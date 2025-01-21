import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv,GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
class CustomGCNConv(MessagePassing):
    def __init__(self):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        return aggr_out
    
class MMPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None, prompt_max_length = 50, n_examples = 3, entity_hidden_size =  768
    ):
        super(MMPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.prompt_max_length = prompt_max_length
        self.n_examples = n_examples
        entity_hidden_size = entity_hidden_size
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        self.kg_encoder = RGCNConv(n_entity, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  
        self.conv_c2 = CustomGCNConv()  
        self.conv_c3 = CustomGCNConv() 
        self.conv_ts1 = CustomGCNConv()  
        self.conv_ts2 = CustomGCNConv()  
        self.conv_ts3 = CustomGCNConv()  
        self.conv_is1 = CustomGCNConv()  
        self.conv_is2 = CustomGCNConv() 
        self.conv_is3 = CustomGCNConv()  
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(None, self.edge_index, self.edge_type)
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = movie_embeds_ts1 + movie_embeds_ts2+ movie_embeds_ts3   
        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = movie_embeds_is1 + movie_embeds_is2 +movie_embeds_is3 
        movie_embeds_mean_t = (movie_embeds_mean_t +movie_embeds_mean_i)/6
        entity_embeds_c1 =self.conv_c1(node_embeds, self.edge_index_c)
        entity_embeds_c2 =self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 =self.conv_c3(entity_embeds_c2, self.edge_index_c)
        entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3+ entity_embeds) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, retrieved_entity_ids = None, word_embeddings = None, mapping = True, context_input_embeddings = None, attention_mask = None):
        batch_size, entity_embeds, entity_len, retrieved_vector, token_len = None, None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds_all = self.get_entity_embeds()
            entity_embeds = entity_embeds_all[entity_ids]  
        if not output_entity:
            assert token_embeds.shape[0] // self.n_examples
            prompt_embeds = token_embeds[:, :self.prompt_max_length, :]
            try:
                prompt_embeds = prompt_embeds.contiguous().view(batch_size, self.n_examples, self.prompt_max_length, self.hidden_size)
                prompt_embeds = prompt_embeds.view(batch_size, self.n_examples * self.prompt_max_length, self.hidden_size)
                pass
            except:
                print(token_embeds.shape)
                print(batch_size, self.n_examples, self.prompt_max_length)
                print(prompt_embeds.shape)
                assert 1==0
            if mapping:
                affinity_scores = self.cross_attn(prompt_embeds) @ word_embeddings.T
                affinity_scores = affinity_scores / self.hidden_size
                prompt_embeds = torch.softmax(affinity_scores, dim =-1) @ word_embeddings
                entity_mean = torch.mean(entity_embeds, dim=1)
                entity_embeds = entity_mean.view(batch_size, self.n_examples * self.prompt_max_length, self.hidden_size)
            prompt_attention_mask = torch.ones((prompt_embeds.shape[0], self.n_examples * self.prompt_max_length)).to(prompt_embeds.device)
            context_input_embeddings = torch.cat([prompt_embeds, context_input_embeddings], dim = 1)
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim =1)
            assert context_input_embeddings.shape[1] == attention_mask.shape[1]
            return context_input_embeddings, attention_mask, retrieved_vector, entity_embeds


    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)




class MMPrompt_inspired(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None,
    ):
        super(MMPrompt_inspired, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        sorted_indices = self.sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        movie_embeds_ts1 = self.conv_ts1(node_features,self.edge_index_t_s)
        movie_embeds_ts2 = self.conv_ts2(movie_embeds_ts1,self.edge_index_t_s)
        movie_embeds_ts3 = self.conv_ts3(movie_embeds_ts2,self.edge_index_t_s)
        movie_embeds_mean_t = (movie_embeds_ts1 +movie_embeds_ts2)/2   
        movie_embeds_is1 = self.conv_is1(node_features,self.edge_index_i_s)
        movie_embeds_is2 = self.conv_is2(movie_embeds_is1,self.edge_index_i_s)
        movie_embeds_is3 = self.conv_is3(movie_embeds_is2,self.edge_index_i_s)
        movie_embeds_mean_i = (movie_embeds_is1+movie_embeds_is2)/2  
        movie_embeds_mean_t =(movie_embeds_mean_t + movie_embeds_mean_i)/2
        entity_embeds_c1 =self.conv_c1(entity_embeds, self.edge_index_c)
        entity_embeds_c2 =self.conv_c2(entity_embeds_c1, self.edge_index_c)
        entity_embeds_c3 =self.conv_c3(entity_embeds_c2, self.edge_index_c)
        entity_embeds = (entity_embeds_c1 + entity_embeds_c2 + entity_embeds_c3+ entity_embeds) / 4
        device = movie_embeds_mean_t.device
        idx_to_id_tensor = self.idx_to_id_tensor.to(device)
        indices = idx_to_id_tensor[:len(movie_embeds_mean_t)]
        entity_embeds.index_add_(0, indices, movie_embeds_mean_t)
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids] 
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        return prompt_embeds











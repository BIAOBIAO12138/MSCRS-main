import json
import os

import torch
from loguru import logger
import json
import os
import tqdm
import torch
from loguru import logger
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import pickle
class DBpedia:
    def __init__(self, dataset, debug=False):
        self.debug = debug

        self.dataset_dir = os.path.join('MSCRS-main/data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self):
        edge_list = set()  
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(self.relation2id)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
        }
        return kg_info
    


class Co_occurrence_inspired:
    def __init__(self, dataset,split, entity_max_length, all_items,n_entity,debug=False):
        self.debug = debug
        self.entity_max_length =entity_max_length
        self.dataset_dir = os.path.join('/MSCRS-main/data', dataset)
        data_file = os.path.join(self.dataset_dir, f'{split}_data_pretrain.jsonl')
        self.co =[]
        self.all_items = set(all_items)
        self.co_occurrence_matrix = np.zeros((n_entity, n_entity))
        self.prepare_data(data_file)


    def prepare_data(self, data_file):        
                input_file = '/MSCRS-main/data/inspired/co_semantic_graph.pkl'
                with open(input_file, 'rb') as f:
                    self.co = pickle.load(f)  
                for transaction in tqdm(self.co):
                    for i in range(len(transaction)):
                        for j in range(i + 1, len(transaction)):
                            product_i = transaction[i]
                            product_j = transaction[j]
                            self.co_occurrence_matrix[product_i][product_j] += 1
                            self.co_occurrence_matrix[product_j][product_i] += 1  
                sparse_matrix = coo_matrix(self.co_occurrence_matrix)
                self.row = sparse_matrix.row
                self.col = sparse_matrix.col
                self.data = sparse_matrix.data
                sparse_matrix = coo_matrix((self.data, (self.row, self.col)), shape=self.co_occurrence_matrix.shape)


                def get_sparse_top_k_similarities(sparse_matrix, k):
                    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(sparse_matrix)
                    distances, indices = nbrs.kneighbors(sparse_matrix)
                    num_entities = sparse_matrix.shape[0]
                    rows = np.repeat(np.arange(num_entities), k)
                    cols = indices.flatten()
                    data = 1 - distances.flatten()  
                    top_k_similarity_matrix = coo_matrix((data, (rows, cols)), shape=(num_entities, num_entities))
                    return top_k_similarity_matrix

                K = 100
                top_k_similarity_matrix = get_sparse_top_k_similarities(sparse_matrix, K)
                self.row_s = top_k_similarity_matrix.row
                self.col_s = top_k_similarity_matrix.col
                self.data_s = top_k_similarity_matrix.data
                new_row_s =[]
                new_col_s=[]
                new_data_s=[]
                for i in range(len(self.row_s)) :
                    if self.col_s[i] not in self.all_items:
                        continue
                    else:
                        new_row_s.append(self.row_s[i])
                        new_col_s.append(self.col_s[i])
                        new_data_s.append(self.data_s[i])

                self.row_s = new_row_s
                self.col_s = new_col_s
                self.data_s = new_data_s
                self.row_s = np.array(self.row_s)
                self.col_s = np.array(self.col_s)
                self.data_s = np.array(self.data_s)
                self.edge_index_s = np.array([self.row_s, self.col_s])
                self.edge_index_s = torch.as_tensor(self.edge_index_s, dtype=torch.long)
                non_one_indices = (self.data != 1) & (self.data != 2)& (self.data != 3)& (self.data != 4)
                self.row_c = self.row[non_one_indices]
                self.col_c = self.col[non_one_indices]
                self.data_c = self.data[non_one_indices]
                self.edge_index_c = np.array([self.row_c, self.col_c])
                self.edge_index_c = torch.as_tensor(self.edge_index_c, dtype=torch.long)
                weight_counts = Counter(self.data_c)
                total_weights = len(self.data_c)
                weight_ratios = {weight: count / total_weights for weight, count in weight_counts.items()}


    def get_entity_co_info(self):
        co_info = {
            'edge_index_s': self.edge_index_s,
            'edge_index_c': self.edge_index_c,
        }
        return co_info



class text_sim:
    def __init__(self,pad_entity_id):

        dataset_dir = '/MSCRS-main/data/inspired'
        data_file = os.path.join(dataset_dir, 'id_embeddings_text.json')
        self.co =[]
        self.pad_entity_id = pad_entity_id
        self.prepare_data(data_file)


    def prepare_data(self, data_file):         

        with open(data_file, 'r', encoding='utf-8') as f:
            id_embeddings = json.load(f)
            new_key = self.pad_entity_id
            new_value = [1.0] * 768  
            id_embeddings[str(new_key)] = new_value
            self.keys = list(id_embeddings.keys())
            self.keys = [int(key) for key in self.keys]
            self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.keys)}
            self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}
            embeddings = np.array(list(id_embeddings.values()))
            similarity_matrix = cosine_similarity(embeddings)
            top_k = 10
            top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:top_k + 1]  # 除去自己
            top_k_dict = {}

            for i, key in enumerate(self.keys):
                top_k_dict[key] = [(self.keys[idx]) for j, idx in enumerate(top_k_indices[i])]
            top_k_dict[new_key] = [new_key]
            mapped_edges = []
            for key, similar_items in top_k_dict.items():
                src_idx = self.id_to_idx[key]
                for target_key in similar_items:
                    tgt_idx = self.id_to_idx[target_key]
                    mapped_edges.append([src_idx, tgt_idx])

            new_list = [[mapped_edges[0][0]], [mapped_edges[0][1]]]

            for i in range(1, len(mapped_edges)):
                new_list[0].append(mapped_edges[i][0])
                new_list[1].append(mapped_edges[i][1])

            self.edge_index_t_s = torch.as_tensor(new_list, dtype=torch.long)


    def get_entity_ts_info(self):
        ts_info = {
            'edge_index_t_s': self.edge_index_t_s,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'all_movie': self.keys,
        }
        return ts_info







class image_sim:
    def __init__(self,pad_entity_id):
        dataset_dir = '/MSCRS-main/data/inspired'
        data_file = os.path.join(dataset_dir, 'id_embeddings_image.json')
        self.co =[]
        self.pad_entity_id = pad_entity_id
        self.prepare_data(data_file)

    def prepare_data(self, data_file):        
        with open(data_file, 'r', encoding='utf-8') as f:
            id_embeddings = json.load(f)
            new_key = self.pad_entity_id
            new_value = [1.0] * 768  
            id_embeddings[str(new_key)] = new_value
            self.keys = list(id_embeddings.keys())
            self.keys = [int(key) for key in self.keys]
            self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.keys)}
            self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}
            embeddings = np.array(list(id_embeddings.values()))
            similarity_matrix = cosine_similarity(embeddings)
            top_k = 10
            top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:top_k + 1]  
            top_k_dict = {}
            for i, key in enumerate(self.keys):
                top_k_dict[key] = [(self.keys[idx]) for j, idx in enumerate(top_k_indices[i])]
            top_k_dict[new_key] = [new_key]
            mapped_edges = []
            for key, similar_items in top_k_dict.items():
                src_idx = self.id_to_idx[key]
                for target_key in similar_items:
                    tgt_idx = self.id_to_idx[target_key]
                    mapped_edges.append([src_idx, tgt_idx])
            new_list = [[mapped_edges[0][0]], [mapped_edges[0][1]]]
            for i in range(1, len(mapped_edges)):
                new_list[0].append(mapped_edges[i][0])
                new_list[1].append(mapped_edges[i][1])
            self.edge_index_i_s = torch.as_tensor(new_list, dtype=torch.long)

    def get_entity_is_info(self):
        is_info = {
            'edge_index_i_s': self.edge_index_i_s,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'all_movie': self.keys,
        }
        return is_info
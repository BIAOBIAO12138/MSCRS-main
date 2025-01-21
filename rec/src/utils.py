from typing import List, Union, Optional

import torch
import wandb
import os

PROJECT_NAME = 'Demonstration-enhanced CRS'
RECOMMENDATION = 'recommendation'
GENERATION = 'generation'
MODEL_NAME = 'UNICRS'

MODEL_RELATED_PARAMS = [
    "n_examples",
    "mapping",
    "prompt_max_length",
    "learning_rate",
    "seed",
    "bias_only",
    "learning_rate"
]


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output




















def convert_params_to_str(params):
    param_str = ""
    for key, value in params.items():
        s = ""
        if key in MODEL_RELATED_PARAMS:
            s = f"[{key}={value}]"
        param_str += s
    return param_str

def init_wandb_run(project_name, dataset, task, tags, model_name, model_params, type_of_run = 'full', run_name = None):

    ### project_name:
    ### task: recommendation or generation
    ### model_name: DCRS or BASElines
    ### model_params: parameters
    ### type of run: full, ablation, analysis.
    if run_name is None:
        run_name = convert_params_to_str(model_params)

    run = wandb.init(project=f"{project_name}",
            group = f"{dataset}-{task}/",
            job_type = type_of_run,
            tags = tags,
            entity="HuyQuangDao",
            reinit=True,
            name = f"{model_name}-{run_name}")

def wandb_logging(eval_dict, step):
    for key, value in eval_dict.items():
        wandb.log(data = {key:value}, step = step)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model_params(gen_model, text_encoder, bias_only = True):
    fix_modules = [text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)
    
    if bias_only:
        #### freeze parameters of the pretrained language model
        for param in gen_model.parameters():
            param.requires_grad = False
        ### only train bias parameters.
        # trainable_components = ['bias']
        # trainable_components = trainable_components + ['pooler.dense.bias']
        # ## unfreeze trainable parameters.
        for para in gen_model.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)
    
        for para in text_encoder.parameters():
            if len(para.shape) <= 1:
                para.requires_grad_(True)

def save(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {k: v for k, v in model.state_dict().items() if 'edge' not in k}
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save(state_dict, save_path)


def load(model, load_dir):
    load_path = os.path.join(load_dir, 'model.pt')
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(load_path, map_location=torch.device('cpu')), strict=False
    )
    return model

import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from config_copy import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv_retrieval_prompt import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia ,Co_occurrence ,text_sim ,image_sim
from evaluate_conv import ConvEvaluator
from utils import init_wandb_run, GENERATION, PROJECT_NAME, MODEL_NAME, wandb_logging, freeze_model_params, count_parameters, save
from model_prompt import MMPrompt
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModel, AutoTokenizer, RobertaForMaskedLM
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=22, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str,default='./prompt-conv-inspired',help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--dataset", type=str, default='inspired', help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int,default=200 ,help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int,default=80, help="max length of decoder input.")
    parser.add_argument("--entity_max_length", type=int,default=64, help="max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=50)
    parser.add_argument("--tokenizer", type=str,default='/home/weiyibiao/weiyibiao/UniCRS-main/src/DialoGPT-small')
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str,default='/home/weiyibiao/weiyibiao/UniCRS-main/src/robert_base')
    parser.add_argument("--model", type=str,default='/home/weiyibiao/weiyibiao/UniCRS-main/src/DialoGPT-small')
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str,default='/home/weiyibiao/weiyibiao/UniCRS-main/src/robert_base')
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int,default=110)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    parser.add_argument("--n_examples", type=int, default=3, help="number of retrieved demonstrations")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=6345)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    parser.add_argument('--mapping', action='store_true', help='if we use semantic mapping')
    parser.add_argument('--bias_only', action='store_true', help='if we use semantic mapping')
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    parser.add_argument('--type_of_run', default='training', help='type of the experiment, eg: full, ablation, analysis')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)
    accelerator = Accelerator(device_placement=False)
    device = accelerator.device
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None
    if args.seed is not None:
        set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    co = Co_occurrence(dataset=args.dataset, split='train', debug=args.debug ,all_items = kg['item_ids'],entity_max_length=args.entity_max_length,n_entity=kg['num_entities'] ).get_entity_co_info()
    text_simi  = text_sim(pad_entity_id=kg['pad_entity_id']).get_entity_ts_info()
    image_simi = image_sim(pad_entity_id=kg['pad_entity_id']).get_entity_is_info()    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)
    print(tokenizer.pad_token_id)
    print(tokenizer.encode('<movie>'))

    prompt_encoder = MMPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],edge_index_c = co['edge_index_c'],edge_index_i_s = image_simi['edge_index_i_s'],edge_index_t_s = text_simi['edge_index_t_s'], idx_to_id = text_simi['idx_to_id'],
        n_prefix_rec=args.n_prefix_conv,
        prompt_max_length = args.prompt_max_length,
        n_examples= args.n_examples
    )

    init_wandb_run(project_name=PROJECT_NAME,
                   dataset = args.dataset,
                   task = GENERATION,
                   model_name=MODEL_NAME,
                   model_params=vars(args),
                   type_of_run=args.type_of_run,
                   tags="Method",
                   run_name=None)
    
    prompt_encoder = prompt_encoder.to(device)
    freeze_model_params(model, text_encoder, bias_only=args.bias_only)
    print('Total numbef of trainable gen params: ', count_parameters(model))
    print('Total numbef of trainable prompt params: ', count_parameters(text_encoder))
    modules = [model, prompt_encoder, text_encoder]
    no_decay = ["bias","LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for m in modules for n, p in m.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for m in modules for n, p in m.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    train_dataset = CRSConvDataset(
        args.dataset, 'train', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    )
    valid_dataset = CRSConvDataset(
        args.dataset, 'valid', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples

    )
    test_dataset = CRSConvDataset(
        args.dataset, 'test', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        n_examples=args.n_examples
    )
    data_collator_teacher = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, use_amp=accelerator.use_fp16, debug=args.debug, gen=False,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length + args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator_teacher,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=accelerator.use_fp16, debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,
        n_examples=args.n_examples,
        prompt_max_length=args.prompt_max_length
    )
    valid_gen_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    test_gen_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    model, prompt_encoder, optimizer, train_dataloader = accelerator.prepare(model, prompt_encoder, optimizer, train_dataloader)

    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'dist@4', 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            #### compute the prompts
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state

            ### compute the retrieval-augmented prompts
            prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=False,
                use_conv_prefix=True,
                mapping = args.mapping,
                # word_embeddings = model.get_input_embeddings()(batch['retrieved_gen']['input_ids']),
                word_embeddings = model.get_input_embeddings().weight,
                context_input_embeddings = model.get_input_embeddings()(batch['context']['input_ids']),
                attention_mask = batch['context']['attention_mask']
            )
            ### re-assign the new computed tensor to the input dictionary
            batch['context']['input_ids'] = None
            ### we directly feed the input embeddings through the generation model
            batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
            batch['context']['attention_mask'] = new_attention_mask

            ## padding the label
            pad_resp = -100 * torch.ones((new_attention_mask.shape[0], args.n_examples * args.prompt_max_length)).to(new_attention_mask.device).long()
            #### padded response
            batch['resp'] = torch.cat([pad_resp, batch['resp']], dim =1)

            loss = model(
                 inputs_embeds = batch['context']['inputs_embeds'],
                 input_ids = None,
                 labels = batch['resp'], 
                 return_dict = True
            )['loss'] / args.gradient_accumulation_steps

            accelerator.backward(loss)
            train_loss.append(float(loss))
            # optim step
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # prompt_optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break
        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        del train_loss, batch

        # dev
        valid_loss = []
        model.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                ### compute the retrieval-augmented prompts
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping = args.mapping,
                    # word_embeddings = model.get_input_embeddings()(batch['retrieved_gen']['input_ids']),
                    word_embeddings = model.get_input_embeddings().weight,
                    context_input_embeddings = model.get_input_embeddings()(batch['context']['input_ids']),
                    attention_mask = batch['context']['attention_mask']
                )
                ### re-assign the new computed tensor to the input dictionary
                # batch['context']['input_ids'] = None
                ### we directly feed the input embeddings through the generation model
                batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
                batch['context']['attention_mask'] = new_attention_mask

                ## padding respose tensor
                pad_resp = -100 * torch.ones((new_attention_mask.shape[0], args.n_examples  * args.prompt_max_length)).to(new_attention_mask.device).long()
                batch['resp'] = torch.cat([pad_resp, batch['resp']], dim =1)

                loss = model(inputs_embeds = batch['context']['inputs_embeds'],
                    input_ids = None,
                    labels = batch['resp'], 
                    return_dict = True
                )['loss']               
                valid_loss.append(float(loss))

        evaluator.log_file.write(f'\n\n*** valid-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(valid_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                ### compute the retrieval-augmented prompts
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping = args.mapping,
                    # word_embeddings = model.get_input_embeddings()(batch['retrieved_gen']['input_ids']),
                    word_embeddings = model.get_input_embeddings().weight,
                    context_input_embeddings = model.get_input_embeddings()(batch['context']['input_ids']),
                    attention_mask = batch['context']['attention_mask']
                )
                ### re-assign the new computed tensor to the input dictionary
                # batch['context']['input_ids'] = None
                ### we directly feed the input embeddings through the generation model
                batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
                batch['context']['attention_mask'] = new_attention_mask
                gen_seqs = accelerator.unwrap_model(model).generate(
                    input_ids = None,
                    inputs_embeds = batch['context']['inputs_embeds'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id.item() for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)
                    # gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)
                
        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_loss = np.mean(valid_loss)
        valid_report['valid/loss'] = valid_loss
        valid_report['epoch'] = epoch
        wandb_logging(eval_dict=valid_report, step = epoch)
        logger.info(valid_report)
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            best_metric = valid_report[f'valid/{metric}']
            # model.save(best_metric_dir)
            save(prompt_encoder, f"{best_metric_dir}/prompt_encoder")
            save(model, f"{best_metric_dir}/gen_model")
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        model.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                ### compute the retrieval-augmented prompts
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping = args.mapping,
                    # word_embeddings = model.get_input_embeddings()(batch['retrieved_gen']['input_ids']),
                    word_embeddings = model.get_input_embeddings().weight,
                    context_input_embeddings = model.get_input_embeddings()(batch['context']['input_ids']),
                    attention_mask = batch['context']['attention_mask']
                )
                ### re-assign the new computed tensor to the input dictionary
                batch['context']['input_ids'] = None
                ### we directly feed the input embeddings through the generation model
                batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
                batch['context']['attention_mask'] = new_attention_mask
            
                pad_resp = -100 * torch.ones((new_attention_mask.shape[0], args.n_examples  * args.prompt_max_length)).to(new_attention_mask.device).long()
                batch['resp'] = torch.cat([pad_resp, batch['resp']], dim =1)

                loss = model(inputs_embeds = batch['context']['inputs_embeds'],
                    input_ids = None,
                    labels = batch['resp'], 
                    return_dict = True
                )['loss']        

                test_loss.append(float(loss))

        evaluator.log_file.write(f'\n*** test-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(test_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
                ### compute the retrieval-augmented prompts
                prompt_augmented_input_embeddings, new_attention_mask, _, _ = prompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True,
                    mapping = args.mapping,
                    # word_embeddings = model.get_input_embeddings()(batch['retrieved_gen']['input_ids']),
                    word_embeddings = model.get_input_embeddings().weight,
                    context_input_embeddings = model.get_input_embeddings()(batch['context']['input_ids']),
                    attention_mask = batch['context']['attention_mask']
                )
                ### re-assign the new computed tensor to the input dictionary
                batch['context']['input_ids'] = None
                ### we directly feed the input embeddings through the generation model
                batch['context']['inputs_embeds'] = prompt_augmented_input_embeddings
                batch['context']['attention_mask'] = new_attention_mask
                gen_seqs = accelerator.unwrap_model(model).generate(
                    input_ids = None,
                    inputs_embeds = batch['context']['inputs_embeds'],
                    max_new_tokens=args.max_gen_len,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)
                    # gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_loss = np.mean(test_loss)
        test_report['test/loss'] = test_loss
        test_report['epoch'] = epoch
        wandb_logging(eval_dict=test_report, step = epoch)
        logger.info(test_report)
        if run:
            run.log(test_report)
        evaluator.reset_metric()

        evaluator.log_cnt += 1

    final_dir = os.path.join(args.output_dir, 'final')
    save(prompt_encoder, f"{final_dir}/prompt_encoder")
    save(model, f"{final_dir}/gen_model")
    logger.info(f'save final model')
    wandb.finish()
import copy
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from pathlib import Path

# 导入 PaddlePaddle 和 Paddlenlp 相关库
import paddle
import paddlenlp
import numpy as np
import datasets
from tqdm import tqdm
from paddle.io import Dataset

# 从 paddlenlp 导入模型、分词器和配置
from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

# --- 辅助 Token ---
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"

# --- Paddle 版的 Tokenizer 和 Embedding Resize 辅助函数 ---
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: paddlenlp.transformers.PretrainedTokenizer,
    model: paddlenlp.transformers.PretrainedModel,
):
    """为分词器添加特殊 token 并相应地调整模型嵌入层的大小。"""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

# --- 基于 PaddlePaddle 的 PPL 计算函数 (滑动窗口) ---
def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, max_length=None, sliding_window_step=8192, truncate=False, aggressive_memory=False
):
    r"""在数据集上计算 "滑动窗口" 困惑度 (PaddlePaddle 实现)"""
    model.eval() # 设置为评估模式

    if add_start_token:
        # 为添加 <BOS> token 留出空间
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]

    pbar = tqdm(total=len(encoded_texts))
    total_nll = paddle.to_tensor(0, dtype='float64')
    total_token_cnt = 0
    for encoding_index in range(0, len(encoded_texts)):
        labels = paddle.to_tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.shape[1]

        prev_end_loc = 0
        eval_length = min(57342, seq_len)
        remain = max(seq_len-eval_length, 1)
        for begin_loc in range(0, remain, sliding_window_step):
            end_loc = min(begin_loc + eval_length, seq_len)
            # print(begin_loc,sliding_window_step, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc]

            if add_start_token:
                bos_tokens_tensor = paddle.to_tensor(
                    [[tokenizer.bos_token_id]] * input_ids.shape[0])
                input_ids = paddle.concat(
                    [bos_tokens_tensor, input_ids], axis=1)

            target_ids = input_ids.clone()
            if trg_len > 0 and trg_len < input_ids.shape[1]:
                target_ids[:, :-trg_len] = -100

            # 3. 执行手动 shift 操作
            #    - 输入：去掉最后一个 token，因为它没有需要预测的下一个 token
            #    - 标签：去掉第一个 token，因为它从不被预测
            input_ids = input_ids[:, :-1]
            target_ids = target_ids[:, 1:]

            with paddle.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                total_nll += neg_log_likelihood * trg_len
                total_token_cnt += trg_len
            
            ppl = float(paddle.exp(total_nll / total_token_cnt).numpy())
            

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.set_postfix(ppl=ppl)
        pbar.update(1)

    ppl = float(paddle.exp(total_nll / total_token_cnt).numpy())
    return {"mean_perplexity": ppl}


def main():
    # 参数解析部分保持不变
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_input_tokens", type=int, default=500)
    parser.add_argument("--max_input_tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--eval_nums", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sliding_window_step", type=int, default=256)
    parser.add_argument('--window_length_list', type=int, nargs='+', default=[])
    parser.add_argument("--truncate", action="store_true", default=False)
    parser.add_argument("--model_max_position_embeddings", type=int, default=2048)
    parser.add_argument("--rope_scaling_factor", type=float, default=1.0)
    parser.add_argument("--rope_scaling_type", type=str, default=None)
    parser.add_argument("--input_field", type=str, default="text")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--path_to_ckp", type=str, default="facebook/llama-7b") # 建议使用HuggingFace模型名称
    parser.add_argument("--dataset_name", type=str, default="scrolls-gov_report")
    parser.add_argument("--path_to_dataset", type=str, default="")
    parser.add_argument("--path_to_output_dir", type=str, default="results/ppls")
    args = parser.parse_args()

    # 设置设备
    paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    print(f"Using device: {paddle.get_device()}")

    model_name_or_path = args.path_to_ckp

    # paddlenlp Llama 模型
    Config, CausalLM, Tokenizer = LlamaConfig, LlamaForCausalLM, AutoTokenizer

    config = Config.from_pretrained(model_name_or_path)
    scaled_max_position_embeddings = int(args.model_max_position_embeddings * args.rope_scaling_factor)
    print(model_name_or_path)

    # paddlenlp 的 LlamaConfig 也支持 rope_scaling 字典
    if not hasattr(config, 'rope_scaling') or config.rope_scaling is None:
        print(type(args.rope_scaling_type))
        if args.rope_scaling_type is not None:
            config.rope_scaling = {"type": args.rope_scaling_type, "factor": args.rope_scaling_factor}
            config.max_position_embeddings = scaled_max_position_embeddings
            if args.rope_scaling_type == "yarn":
                config.rope_scaling["original_max_position_embeddings"] = args.model_max_position_embeddings
            
    config.use_cache = False
    config.use_flash_attention = True

    # 加载 Paddle 模型，使用 dtype='float16'
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, 
        config=config,
        dtype='float16' 
    )

    tokenizer = Tokenizer.from_pretrained(model_name_or_path)
    
    # 使用 Paddle 版的辅助函数
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
    if "llama" in args.model_name:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    
    print(f"dataset:{args.dataset_name}")
    if "scrolls" in args.dataset_name:
        args.input_field = "input"
    elif "pile" in args.dataset_name:
        args.input_field = "text"
    elif "proof" in args.dataset_name:
        args.input_field = "text"

    input_texts = datasets.load_dataset("json", data_files=args.path_to_dataset, split="train")

    def tokenize(example):
        tokenized = tokenizer(
            example[args.input_field],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens - 1, # 为 <BOS> token 留出空间
            return_attention_mask=True,
        )
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]
        example["tokenized_len"] = len(tokenized["input_ids"])
        return example

    input_texts = input_texts.map(tokenize, num_proc=128)

    if args.min_input_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.min_input_tokens - 1)
    if args.eval_nums:
        input_texts = input_texts.select(range(args.eval_nums)) # 使用 .select() 更高效

    ppl_list = []
    context_window_size = args.window_length_list
    print(context_window_size)

    for ctx_size in context_window_size:
        ppl = compute_perplexity(encodings=input_texts, model=model, tokenizer=tokenizer, add_start_token=True, max_length=ctx_size, truncate=args.truncate)["mean_perplexity"]
        # ppl = compute_perplexity(encodings=input_texts, model=model, tokenizer=tokenizer, add_start_token=True, max_length=ctx_size, sliding_window_step=args.sliding_window_step, truncate=args.truncate)["mean_perplexity"]

        print(f"context window size: {ctx_size}; ppl: {ppl}")
        
        ppl_list.append(ppl)



if __name__ == "__main__":
    main()
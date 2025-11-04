# debug_mode="-m debugpy --listen 127.0.0.1:6679 --wait-for-client"
export CUDA_VISIBLE_DEVICES=6

llama_prefix=facebook/llama-7b                                        # LLaMA-7B
# llama_prefix=/data/whf/ssa/checkpoints/llama-7b/64k/checkpoint-600      # SSA - LLaMA-7B
data_prefix=/home/whf/PoSE/PoSE-Datasets

python ${debug_mode} src/ppl.py \
    --path_to_ckp ${llama_prefix} \
    --model_max_position_embeddings 2048 \
    --max_input_tokens 81920 \
    --min_input_tokens 81920 \
    --window_length_list 49152 65536 81920 \
    --truncate \
    --dataset_name proof-pile \
    --path_to_dataset ${data_prefix}/proof-pile/test.jsonl \
    --rope_scaling_type yarn \
    --rope_scaling_factor 32



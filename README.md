# 仓库
* Segmented Sparse Attention方法的Paddle实现
# SSA长文预训练
* nohup python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" src/run_finetune.py src/ssa_arguments.json > train_log/ssa_train_test.log 2>&1 &
# 长文推理
* bash scripts/run_eval_ppl.sh
# 依赖环境
* Paddle 3.3.0
* PaddleNLP 3.0.0b4

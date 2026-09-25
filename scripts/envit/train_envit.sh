export PYTHONPATH=$(pwd)
export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_NAME="Ekanari/Adobe_5K"

python src/ENVIT/ENVIT_trainer.py \
    --devices 0  \
    --do_train \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 5 \
    --precision bf16-mixed \
    --resolution 224 \
    --patch_size 7 \
    --dim 1024 \
    --depth 6 \
    --dim_head 64 \
    --heads 8 \
    --mlp_dim 2048 \
    --emb_dim 32 \
    --dropout 0.15 \
    --emb_dropout 0.15 \
    --train_batch_size 100 \
    --learning_rate 5e-4 \
    --num_train_epochs 2 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --save_ckpt_path "checkpoint/ENVIT_stage1" \
    --output_dir "checkpoint/ENVIT_stage1" \
    --wandb_run_name "ENVIT_stage1"

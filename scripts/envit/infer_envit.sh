export PYTHONPATH=$(pwd)

python src/ENVIT/inference.py \
    --devices 0  \
    --do_train_stage2 \
    --do_eval \
    --resolution 224 \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 2 \
    --resolution 224 \
    --precision bf16-mixed \
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
    --learning_rate 1e-6 \
    --num_train_epochs 2 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --save_ckpt_path "checkpoint/ENVIT_stage2" \
    --output_dir "checkpoint/ENVIT_stage2" \
    --wandb_run_name "ENVIT_stage2"

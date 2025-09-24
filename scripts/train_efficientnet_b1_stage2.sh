export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_NAME="Ekanari/Adobe_5K"

python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train_stage2 \
    --save_weights_only \
    --num_label 2 \
    --val_batch_size 32 \
    --dataloader_num_workers 8 \
    --precision bf16-mixed \
    --resolution 256 \
    --learning_rate 1e-6 \
    --num_train_epochs 2 \
    --adam_weight_decay 5e-4 \
    --gradient_accumulation_steps 1 \
    --save_ckpt_path "checkpoint/Deepfake_Detection_no_encoder_stage2" \
    --output_dir "checkpoint/Deepfake_Detection_no_encoder_stage2" \
    --wandb_run_name "Deepfake_Detection_no_encoder_stage2"

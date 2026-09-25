#--get_first_dim \

python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 5 \
    --train_json_file "data/audio_labels_mel.json" \
    --precision bf16 \
    --train_batch_size 100 \
    --mode efficientnet \
    --learning_rate 5e-4 \
    --num_train_epochs 50 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "None" \
    --save_ckpt_path "checkpoint/Efficientnetb1_Mel_stage1_3loss_100epoch" \
    --output_dir "checkpoint/Efficientnetb1_Mel_stage1_3loss_100epoch" \
    --wandb_run_name "Efficientnetb1_Mel_stage1"

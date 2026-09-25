 #   --get_first_dim \



python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train_stage2 \
    --do_eval \
    --save_weights_only \
    --num_label 2 \
    --train_batch_size 72 \
    --val_batch_size 32 \
    --dataloader_num_workers 8 \
    --train_json_file "data/audio_labels_mel_dev.json" \
    --val_json_file "data/audio_labels_mel_test.json" \
    --precision bf16 \
    --save_top_k 10 \
    --mode efficientnet \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Efficientnetb1_Mel_stage3_5epoch/checkpoint/best.pt" \
    --save_ckpt_path "checkpoint/Efficientnetb1_Mel_stage4_5epoch_new/" \
    --output_dir "checkpoint/Efficientnetb1_Mel_stage4_5epoch_new/checkpoint" \
    --wandb_run_name "Efficientnetb1_Mel_stage4" 

# oh my, the mel have save into the gam, guess i have to train gam again
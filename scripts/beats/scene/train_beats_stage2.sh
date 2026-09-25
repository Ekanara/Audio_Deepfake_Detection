 #   --get_first_dim \



python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train_stage2 \
    --save_weights_only \
    --num_label 2 \
    --train_batch_size 72 \
    --val_batch_size 32 \
    --dataloader_num_workers 8 \
    --train_json_file "data/label/beats/audio_labels_beats_stage2.json" \
    --precision bf16-mixed \
    --mixup \
    --save_top_k 10 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --adam_weight_decay 0 \
    --mode beats \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Beats/Beats_stage1_3loss_50epoch/checkpoint/best.pt" \
    --save_ckpt_path "checkpoint/Beats/Beats_stage2_10_epoch" \
    --output_dir "checkpoint/Beats/Beats_stage2_10_epoch/checkpoint" \
    --wandb_run_name "Beats_stage2" 

# oh my, the mel have save into the gam, guess i have to train gam again
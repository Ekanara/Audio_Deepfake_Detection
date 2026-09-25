 #   --get_first_dim \



python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train_stage2 \
    --save_weights_only \
    --num_label 2 \
    --do_eval \
    --prev_num_label 5 \
    --train_batch_size 72 \
    --val_batch_size 32 \
    --dataloader_num_workers 8 \
    --train_json_file "data/label/beats/audio_labels_beats_event_stage3.json" \
    --val_json_file "data/label/beats/test_track2.json" \
    --precision bf16 \
    --save_top_k 3 \
    --learning_rate 8e-7 \
    --num_train_epochs 5 \
    --adam_weight_decay 0 \
    --mode beats \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Beats/Beats_3stage/Beats_event_stage1_60_epoch/checkpoint/best.pt" \
    --save_ckpt_path "checkpoint/Beats/Beats_3stage/Beats_event_stage3_5_epoch_for_now_new" \
    --output_dir "checkpoint/Beats/Beats_3stage/Beats_event_stage3_5_epoch_for_now_new/checkpoint" \
    --wandb_run_name "Beat_event_stage3" 

# oh my, the mel have save into the gam, guess i have to train gam again
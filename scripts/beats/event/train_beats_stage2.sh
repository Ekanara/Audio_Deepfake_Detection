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
    --prev_num_label 5 \
    --train_batch_size 9 \
    --val_batch_size 32 \
    --dataloader_num_workers 8 \
    --train_json_file "data/label/beats/Event_train.json" \
    --val_json_file "data/label/beats/test_track2.json" \
    --precision bf16 \
    --save_top_k 30 \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    --adam_weight_decay 0 \
    --mode beats \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Beats_journal/Beats_Event_stage1_3loss_50epoch_batch30/checkpoint/best.pt" \
    --save_ckpt_path "checkpoint/Beats_journal/Beats_Event_stage2_10epoch_batch30_no_mixup" \
    --output_dir "checkpoint/Beats_journal/Beats_Event_stage2_10epoch_batch30/checkpoint" \
    --wandb_run_name "Beats_TUTSED16_stage2" 

# oh my, the mel have save into the gam, guess i have to train gam again
#--do_eval \
#    --mixup \

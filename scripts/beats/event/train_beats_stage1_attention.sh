#--get_first_dim \

python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train_attention \
    --three_loss \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 5 \
    --train_json_file "data/label/beats/Event_train_stage1.json" \
    --precision bf16 \
    --train_batch_size 5 \
    --mode beats \
    --learning_rate 5e-4 \
    --num_train_epochs 10 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Beats/Beats_Event_5e-6/checkpoint/best.pt" \
    --save_ckpt_path "checkpoint/Beats_journal/Beats_Event_stage1_3loss_50epoch_batch30_freeze_backbone" \
    --output_dir "checkpoint/Beats_journal/Beats_Event_stage1_3loss_50epoch_batch30_freeze_backbone" \
    --wandb_run_name "Beats_event_stage1"

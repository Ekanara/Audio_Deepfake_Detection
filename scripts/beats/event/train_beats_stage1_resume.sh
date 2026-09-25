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
    --train_json_file "data/label/beats/audio_labels_beats_event_stage1.json" \
    --precision bf16 \
    --train_batch_size 5 \
    --mode beats \
    --learning_rate 1e-6 \
    --num_train_epochs 50 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "checkpoint/Beats/Beats_3stage/Beats_event_stage1_3loss_50epoch" \
    --save_ckpt_path "checkpoint/Beats/Beats_3stage/Beats_event_stage1_3loss_50epoch" \
    --output_dir "checkpoint/Beats/Beats_3stage/Beats_event_stage1_3loss_60epoch" \
    --wandb_run_name "Beats_event_stage1"

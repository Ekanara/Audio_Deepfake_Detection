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
    --train_json_file "data/label/beats/audio_labels_beats_all_stage1.json" \
    --precision bf16 \
    --train_batch_size 100 \
    --mode beats \
    --learning_rate 5e-6 \
    --num_train_epochs 50 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "None" \
    --save_ckpt_path "checkpoint/Beats_3stage/Beats_all_stage1_3loss_50epoch" \
    --output_dir "checkpoint/Beats_3stage/Beats_all_stage1_3loss_50epoch" \
    --wandb_run_name "Beats_all_stage1"

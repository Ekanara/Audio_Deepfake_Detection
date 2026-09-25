#--get_first_dim \

python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 2 \
    --train_json_file "data/label/gam/audio_labels_gam_event.json" \
    --precision bf16 \
    --train_batch_size 100 \
    --mode densenet \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "None" \
    --save_ckpt_path "checkpoint/Densenet_161/Densenet161_Gam_event_bench" \
    --output_dir "checkpoint/Densenet_161/Densenet161_Gam_event_bench/checkpoint" \
    --wandb_run_name "Densenet_161_event_bench"

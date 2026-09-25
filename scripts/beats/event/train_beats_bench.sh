#--get_first_dim \

python src/base_trainer.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name_train=$DATASET_NAME \
    --dataset_name_validation=$DATASET_NAME \
    --devices 0  \
    --do_train \
    --do_eval \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 2 \
    --train_json_file "data/label/beats/TUTSED16_train.json" \
    --val_json_file "data/label/beats/test_track2.json" \
    --precision bf16 \
    --train_batch_size 5 \
    --mode beats \
    --learning_rate 5e-6  \
    --num_train_epochs 10 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "None" \
    --save_ckpt_path "checkpoint/Beats_journal/Beats_TUTSED16_5e-6_bench" \
    --output_dir "checkpoint/Beats_journal/Beats_TUTSED16_5e-6_bench" \
    --wandb_run_name "Beats_Event"

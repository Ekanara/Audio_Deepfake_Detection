#--get_first_dim \

python src/base_trainer.py \
    --devices 0  \
    --do_train \
    --three_loss \
    --save_weights_only \
    --dataloader_num_workers 8 \
    --num_label 5 \
   --beats_feature encoder \
    --embed_dim 768\
    --train_json_file "data/label/beats/Event_train_stage1.json" \
    --val_json_file "data/label/beats/test_track2.json" \
    --precision bf16 \
    --train_batch_size 5 \
    --mode beats \
    --learning_rate 3e-7 \
    --num_train_epochs 5 \
    --adam_weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --load_ckpt_path "None" \
    --save_ckpt_path "checkpoint/Beats_journal/Beats_Event_527_realfocus_stage1_0.8cclloss_0.2arcfaceloss_5epoch_batch30_3e-7" \
    --output_dir "checkpoint/Beats_journal/Beats_Event_527_realfocus_stage1_0.8cclloss_0.2arcfaceloss_5epoch_batch30_3e-7" \
    --wandb_run_name "Beats_event_stage1"

export PYTHONPATH=$(pwd)

python src/inference/inference_ccl.py \
    --devices 0 \
    --resolution 224 \
    --mode beats \
    --ref_json_file "data/label/beats/old/Event_real.json" \
    --beats_feature encoder \
    --val_json_file "data/label/beats/test_track2.json" \
    --load_ckpt_path "checkpoint/Beats_journal/Beats_Event_768_realfocus_stage1_0.8cclloss_0.2arcfaceloss_5epoch_batch30_3e-7/checkpoint/best.pt" \
    --output_dir runs/ccl_event_768_realfocui
export PYTHONPATH=$(pwd)
    #--get_first_dim \

python src/inference/inference.py \
    --devices 0  \
    --resolution 224 \
    --mode beats \
    --val_json_file "data/label/beats/test_track2.json" \
    --load_ckpt_path "checkpoint/Beats_3stage/Beats_all_stage3_5_epoch/checkpoint/best.pt"

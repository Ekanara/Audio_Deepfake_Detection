export PYTHONPATH=$(pwd)
    #--get_first_dim \

python src/inference/inference.py \
    --devices 0  \
    --resolution 224 \
    --mode densenet \
    --val_json_file "data/label/gam/audio_labels_gam_scene_test.json" \
    --load_ckpt_path "checkpoint/Densenet_161/Densenet161_Gam_event_bench/checkpoint/best.pt"

export PYTHONPATH=$(pwd)
    #--get_first_dim \

python src/inference/inference.py \
    --devices 0  \
    --resolution 224 \
    --mode inception \
    --val_json_file "data/label/gam/audio_labels_gam_event_test.json" \
    --load_ckpt_path "checkpoint/Inception_v3/Inceptionv3_Gam_scene_bench/checkpoint/best.pt"

export PYTHONPATH=$(pwd)
    #--get_first_dim \

python src/inference/inference.py \
    --devices 0  \
    --resolution 224 \
    --mode efficientnet \
    --val_json_file "data/label/cqt/audio_labels_cqt_scene_test.json" \
    --load_ckpt_path "checkpoint/Efficientnet_b1/Efficientnetb1_Cqt_scene_bench/checkpoint/best.pt"

export PYTHONPATH=$(pwd)
    #--get_first_dim \

# python src/inference/inference.py \
#     --devices 0  \
#     --mode resnet \
#     --resolution 224 \
#     --val_json_file "data/label/mel/Clotho_test.json" \
#     --load_ckpt_path "checkpoint/Resnet_50/Resnet50_Mel_TUTSED16.json/checkpoint/best.pt"


python src/inference/inference.py \
    --devices 0  \
    --mode resnet \
    --resolution 224 \
    --val_json_file "data/label/cqt/TUTSED16_test.json" \
    --load_ckpt_path "checkpoint/Resnet_50/Resnet50_Cqt_TUTSED16_train.json/checkpoint/best.pt"
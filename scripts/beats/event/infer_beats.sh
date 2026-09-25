export PYTHONPATH=$(pwd)
    #--get_first_dim \

python src/inference/inference.py \
    --devices 0  \
    --resolution 224 \
    --mode beats \
    --val_json_file "data/label/beats/test_track2.json" \
    --load_ckpt_path "checkpoint/Beats_journal/Beats_TUTSED16_stage2_30epoch_batch30_new_no_mixup/checkpoint/sample-epoch=10-valid/loss=0.51.ckpt"

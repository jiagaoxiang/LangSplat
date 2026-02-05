#!/bin/bash
CASE_NAME="figurines"

# path to lerf_ovs/label
gt_folder="/home/douglas.jia@amd.com/LangSplat/lerf_ovs/label"

root_path="../"

# Base model path: same as -m used in train/render commands.
# The eval script appends _{1,2,3} for the three feature levels.
# Examples:
#   For single-GPU training with: -m output/figurines       -> set MODEL_PATH=output/figurines
#   For DDP training with:        -m output/figurines_ddp   -> set MODEL_PATH=output/figurines_ddp
MODEL_PATH="${root_path}/output/${CASE_NAME}_ddp"

python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --model_path ${MODEL_PATH} \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --json_folder ${gt_folder}

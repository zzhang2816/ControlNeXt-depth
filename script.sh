# conda activate svd_controlnet

export NCCL_P2P_DISABLE=1
export PATH_TO_THE_GROUND_TRUTH_DIR_FOR_EVALUATION="/data2/zhiyuan/depth_validation/"
export PATH_TO_THE_SAVE_DIR="/data2/zhiyuan/controlnext_debug"
export PATH_TO_CSV="/data2/zhiyuan/WebVid10K/shuffled_file.csv"
export PATH_TO_VIDEO="/data2/wff/webvid"
export PATH_TO_DEPTH="/data2/zhiyuan/WebVid10K/depth"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./deepspeed.yaml train_svd.py \
 --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid \
 --output_dir=$PATH_TO_THE_SAVE_DIR \
 --csv_path=$PATH_TO_CSV \
 --video_folder=$PATH_TO_VIDEO \
 --condition_folder=$PATH_TO_DEPTH \
 --dataset_type="webvid10k" \
 --validation_image_folder=$PATH_TO_THE_GROUND_TRUTH_DIR_FOR_EVALUATION \
 --width=512 \
 --height=320 \
 --lr_warmup_steps 500 \
 --sample_n_frames 14 \
 --learning_rate=1e-5 \
 --per_gpu_batch_size=2 \
 --num_train_epochs=10 \
 --mixed_precision="bf16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=2000 \
 --validation_steps=1000 \
 --gradient_checkpointing \
 --checkpoints_total_limit 4 

# # For Resume
#  --controlnet_model_name_or_path $PATH_TO_THE_CONTROLNEXT_WEIGHT
#  --unet_model_name_or_path $PATH_TO_THE_UNET_WEIGHT




#  --meta_info_path=$PATH_TO_THE_META_INFO_FILE_FOR_DATASET \
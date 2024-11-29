# conda activate df_latest
export pretrained="/data2/zhiyuan/controlnext_refined/checkpoint-last"
CUDA_VISIBLE_DEVICES=0 python run_controlnext.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid \
  --output_dir outputs \
  --guidance_scale 1.5 \
  --batch_frames 14 \
  --sample_stride 1 \
  --overlap 6 \
  --height 320 \
  --width 512 \
  --controlnext_path ${pretrained}/controlnext/diffusion_pytorch_model.bin \
  --unet_path ${pretrained}/unet/diffusion_pytorch_model.bin \
  --validation_control_video_path 023_depth.mp4 \
  --ref_image_path 00000.png


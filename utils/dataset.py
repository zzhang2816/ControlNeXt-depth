import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader, cpu
import cv2
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from utils.util import zero_rank_print
#from torchvision.io import read_image
from PIL import Image
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB').resize((512, 320)) # tmp fix
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,depth_folder,motion_folder,
            sample_size=256, sample_stride=4, sample_n_frames=14,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.depth_folder = depth_folder
        self.motion_values_folder=motion_folder
        print("length",len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size",sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    




    def center_crop(self,img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top:top+min_dim, left:left+min_dim]
        
    
    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[1].split('.')[0])
    

    
        while True:
            video_dict = self.dataset[idx]
            videoid = video_dict['videoid']
    
            preprocessed_dir = os.path.join(self.video_folder, videoid)
            depth_folder = os.path.join(self.depth_folder, videoid)
            motion_values_file = os.path.join(self.motion_values_folder, videoid, videoid + "_average_motion.txt")
    
            if not os.path.exists(depth_folder) or not os.path.exists(motion_values_file):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Sort and limit the number of image and depth files to 14
            image_files = sorted(os.listdir(preprocessed_dir), key=sort_frames)[:14]
            depth_files = sorted(os.listdir(depth_folder), key=sort_frames)[:14]
    
            # Check if there are enough frames for both image and depth
            if len(image_files) < 14 or len(depth_files) < 14:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
    
            # Load depth frames
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_files])
            depth_pixel_values = numpy_to_pt(numpy_depth_images)
    
            # Load motion values
            with open(motion_values_file, 'r') as file:
                motion_values = float(file.read().strip())
    
            return pixel_values, depth_pixel_values, motion_values

        
        
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        #while True:
           # try:
        pixel_values, depth_pixel_values,motion_values = self.get_batch(idx)
           #     break
          #  except Exception as e:
          #      print(e)
          #      idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values,motion_values=motion_values)
        return sample



class WebVidCustom(Dataset):
    def __init__(
            self,
            video_folder, depth_folder,
            sample_size=256, sample_n_frames=14,
        ):

        self.video_ids = os.listdir(depth_folder)[:1000]
        self.length = len(self.video_ids)
        print(f"data scale: {self.length}")
        self.video_folder = video_folder
        self.depth_folder = depth_folder
        self.sample_n_frames = sample_n_frames
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size),
            transforms.CenterCrop(sample_size),
            transforms.Resize((320, 512)), # tmp fix
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        def get_frame_i(frame_name):
            return int(frame_name.split('.')[0])
    
        videoid = self.video_ids[idx]

        video_path = os.path.join(self.video_folder, videoid+".mp4")
        depth_folder = os.path.join(self.depth_folder, videoid)
        depth_frames = os.listdir(depth_folder)
        if len(depth_frames) < self.sample_n_frames:
            return None
        
        depth_frames = sorted(os.listdir(depth_folder), key=get_frame_i)[:self.sample_n_frames]
        batch_index = [get_frame_i(frame) for frame in depth_frames]
        
        # Load image frames
        video_reader = VideoReader(video_path)
        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        # Load depth frames
        numpy_depth_images = np.array([pil_image_to_numpy(Image.open(os.path.join(depth_folder, df))) for df in depth_frames])
        depth_pixel_values = numpy_to_pt(numpy_depth_images)

        return pixel_values, depth_pixel_values
     
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, depth_pixel_values = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        reference_image = pixel_values[0]
        sample = dict(pixel_values=pixel_values, guide_values=depth_pixel_values, reference_image=reference_image)
        return sample


class WebVid10KDepth(Dataset):
    def __init__(self,
                 csv_path,
                 video_folder,
                 depth_folder,
                 motion_folder,
                 subsample=None,
                 sample_n_frames=14,
                 sample_size=[320, 512],
                 sample_stride=4,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
        self.meta_path = csv_path
        self.video_folder = video_folder
        self.depth_folder = depth_folder
        self.subsample = subsample
        self.video_length = sample_n_frames
        self.resolution = [sample_size, sample_size] if isinstance(sample_size, int) else sample_size
        self.fps_max = fps_max
        self.frame_stride = sample_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.video_ids, self.captions = self.load_vids_captions()

        self.spatial_transform = transforms.Compose([
            transforms.Resize(self.resolution[1]), # this is a bug, should be resolution[0]
            transforms.CenterCrop(self.resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
                
    
    def load_vids_captions(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        
        video_ids = metadata['videoid'].tolist()
        captions = metadata['name'].tolist()
        return video_ids, captions

    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            # index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            # video_path = self._get_video_path(sample)
            # ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            videoid = self.video_ids[index]
            caption = self.captions[index]
            video_path = os.path.join(self.video_folder, f"{videoid}.mp4")
            depth_folder = os.path.join(self.depth_folder, f"{videoid}")

            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = min(len(video_reader), 200)

            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]

            ## retrieve the depth map 
            depth_values = np.stack([np.array(Image.open(f"{depth_folder}/{idx}.png").convert('RGB')) for idx in frame_indices])
            depth_values = torch.from_numpy(depth_values).permute(0, 3, 1, 2)
            depth_values = depth_values / 255

            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = frames / 255.
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        reference_image = frames[0]
        data = {'pixel_values': frames, 'guide_values': depth_values, 'reference_image': reference_image}
        return data
    
    def __len__(self):
        return len(self.video_ids)




if __name__ == "__main__":
    from utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/data/webvid/results_2M_train.csv",
        video_folder="/data/webvid/data/videos",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)
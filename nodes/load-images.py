import os
from PIL import Image, ImageOps, ImageStat
import numpy as np
import torch
import functools

class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 1024, "min": 64, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 64, "step": 1}),
                "keep_aspect_ratio": (["crop", "pad", "stretch","no_resize"],), 
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING",)
    RETURN_NAMES = ("image", "count", "image_path",)

    FUNCTION = "load_images"
    CATEGORY = "KAndy"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

  
    def _glob_files(self, folder, include_subfolders=False):
        
        @functools.cache
        def _int_glob_files(folder, include_subfolders=False):
            """Return a list of image paths from the given folder. Use cache for speed"""
            
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_paths = []
            
            if include_subfolders:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in valid_extensions):
                            image_paths.append(os.path.join(root, file))
            else:
                for file in os.listdir(folder):
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(folder, file))

            image_paths = sorted(image_paths)
            return image_paths
        
        return _int_glob_files(folder, include_subfolders)
        
    def load_images(self, folder, width, height, image_load_cap, start_index, keep_aspect_ratio, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        
        dir_files = self._glob_files(folder, include_subfolders)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            # Resize image to maximum dimensions, if needed
            i = self.resize_with_aspect_ratio(i, width, height, keep_aspect_ratio)
            
            
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            images.append(image)
            image_path_list.append(image_path)
            image_count += 1

        if len(images) == 1:
            return (images[0], 1, image_path_list)
        
        elif len(images) > 1:
            image1 = images[0]
            for image2 in images[1:]:
                image1 = torch.cat((image1, image2), dim=0)

            return (image1, len(images), image_path_list)
        

    def resize_with_aspect_ratio(self, img, width, height, mode):
        if mode == "no_resize" or img.size == (width, height):
            return img
        
        if mode == "stretch":
            return img.resize((width, height), Image.Resampling.LANCZOS)
        
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        target_ratio = width / height

        if mode == "crop":
            # Calculate dimensions for center crop
            if aspect_ratio > target_ratio:
                # Image is wider - crop width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                left = (new_width - width) // 2
                return img.crop((left, 0, left + width, height))
            else:
                # Image is taller - crop height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - height) // 2
                return img.crop((0, top, width, top + height))

        elif mode == "pad":
            pad_color = self.get_edge_color(img)
            # Calculate dimensions for padding
            if aspect_ratio > target_ratio:
                # Image is wider - pad height
                new_height = int(width / aspect_ratio)
                img = img.resize((width, new_height), Image.Resampling.LANCZOS)
                padding = (height - new_height) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (0, padding))
                return padded
            else:
                # Image is taller - pad width
                new_width = int(height * aspect_ratio)
                img = img.resize((new_width, height), Image.Resampling.LANCZOS)
                padding = (width - new_width) // 2
                padded = Image.new('RGBA', (width, height), pad_color)
                padded.paste(img, (padding, 0))
                return padded
            
    def get_edge_color(self, img):
        """Sample edges and return dominant color"""
        width, height = img.size
        img = img.convert('RGBA')
        
        # Create 1-pixel high/wide images from edges
        top = img.crop((0, 0, width, 1))
        bottom = img.crop((0, height-1, width, height))
        left = img.crop((0, 0, 1, height))
        right = img.crop((width-1, 0, width, height))
        
        # Combine edges into single image
        edges = Image.new('RGBA', (width*2 + height*2, 1))
        edges.paste(top, (0, 0))
        edges.paste(bottom, (width, 0))
        edges.paste(left.resize((height, 1)), (width*2, 0))
        edges.paste(right.resize((height, 1)), (width*2 + height, 0))
        
        # Get median color
        stat = ImageStat.Stat(edges)
        median = tuple(map(int, stat.median))
        return median
    
__NODE__ = LoadImagesFromFolder
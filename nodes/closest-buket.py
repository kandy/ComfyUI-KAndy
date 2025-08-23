import numpy as np

class HyVideoGetClosestBucketSize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    "base_size": (["360", "540", "720"], {"default": "540", "tooltip": "Resizes the input image to closest original training bucket size"}),
                    },
                }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "encode"
    CATEGORY = "Kandy"

    def encode(self, image, base_size):
        if base_size == "720":
            bucket_hw_base_size = 960
        elif base_size == "540":
            bucket_hw_base_size = 720
        elif base_size == "360":
            bucket_hw_base_size = 480
        else:
           return (base_size, base_size,)
        B, H, W, C = image.shape
        crop_size_list = self.generate_crop_size_list(int(bucket_hw_base_size), 32)
        aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = self.get_closest_ratio(H, W, aspect_ratios, crop_size_list)
       
        return (closest_size[1], closest_size[0],)
    
    def generate_crop_size_list(self, base_size=256, patch_size=16, max_ratio=4.0):
        num_patches =  round((base_size / patch_size) ** 2)
        assert max_ratio >= 1.
        crop_size_list = []
        wp, hp = num_patches, 1
        while wp > 0:
            if max(wp, hp) / min(wp, hp) <= max_ratio:
                crop_size_list.append((wp * patch_size, hp * patch_size))
            if (hp + 1) * wp <= num_patches:
                hp += 1
            else:
                wp -= 1
        return crop_size_list
    def get_closest_ratio(self, height: float, width: float, ratios: list, buckets: list):
        aspect_ratio = float(height)/float(width)
        closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
        closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
        return buckets[closest_ratio_id], float(closest_ratio)
    
__NODE__= HyVideoGetClosestBucketSize
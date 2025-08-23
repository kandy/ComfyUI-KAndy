import requests
from PIL import Image, ImageOps
import torch
import numpy as np

class KAndyLoadImageFromUrl:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load"
    CATEGORY = "kandy"

    def load(self, url):
        try:
            # get the image from the url
            headers = {
                "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "accept-language": "en,en-US;q=0.9",
                "priority": "i",
                "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "image",
                "sec-fetch-mode": "no-cors",
                "sec-fetch-site": "cross-site",  
                "referrer": "https://www.reddit.com/",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
            }
            for i in range(3):
                try:
                    image = Image.open(requests.get(url, stream=True, headers=headers, timeout=3).raw)
                    break
                except requests.exceptions.Timeout:
                    if i == 2:
                        return (None, "Timeout Error")
                    else:
                        continue
            image = ImageOps.exif_transpose(image)
            return (self.pil2tensor(image), "Success")
        except Exception as e:
            return (None, f"Error: {str(e)}")

    def pil2tensor(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        """Converts a PIL Image or a list of PIL Images to a tensor."""

        def single_pil2tensor(image: Image.Image) -> torch.Tensor:
            np_image = np.array(image).astype(np.float32) / 255.0
            if np_image.ndim == 2:  # Grayscale
                return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
            else:  # RGB or RGBA
                return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

        if isinstance(images, Image.Image):
            return single_pil2tensor(images)
        else:
            return torch.cat([single_pil2tensor(img) for img in images], dim=0)

__NODE__ = KAndyLoadImageFromUrl
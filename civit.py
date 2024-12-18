import requests
import urllib


class KCivitPromptAPI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit": ("INT", {"default": 0, "min": 0, "max": 128000, "step": 1},)
            },
            "optional": { 
                "cursor": ("STRING",),
                "postId": ("STRING",),
                "modelId": ("STRING",),
                "username": ("STRING",),
                "period": (["AllTime", "Year", "Month", "Week", "Day", ""],), 
                "nsfw": (["Soft", "Mature", "X", ""],),
                "sort": (["Most Reactions", "Most Comments","Newest", ""],) 
            }
        }

    RETURN_NAMES = ("URLs","cursor")
    RETURN_TYPES = ("STRING", "STRING",)
    
    CATEGORY = "kandy"
    
    FUNCTION = "ffetch"
    
    def ffetch(self, **kwargs):
        url = "https://civitai.com/api/v1/images"
        urls = []
        params = {k: v for k, v in kwargs.items() if v}
        url = '{}?{}'.format(url, urllib.parse.urlencode(params))
        print(url)
        response = requests.get(url)

        if response.status_code == 200:
            body = response.json()
            prompts = [item.get("meta").get("prompt") for item in body.get("items") if item.get("meta") and item.get("meta").get("prompt")] 
            # print(prompts)
            return (prompts, body.get("metadata").get("nextCursor"),)
        else:
            print(response.status_code)

        return (urls,"",)


class KCivitImagesAPI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit": ("INT", {"default": 0, "min": 0,  "step": 1},)
            },
            "optional": { 
                "cursor": ("STRING",),
                "postId": ("STRING",),
                "modelId": ("STRING",),
                "username": ("STRING",),
                "period": (["AllTime", "Year", "Month", "Week", "Day", ""],), 
                "nsfw": (["Soft", "Mature", "X", ""],),
                "sort": (["Most Reactions", "Most Comments","Newest", ""],) 
            }
        }

    RETURN_NAMES = ("IMAGES", "Cursor",)
    RETURN_TYPES = ("STRING","STRING",)

    FUNCTION = "ffetch"
    CATEGORY = "kandy"

    def ffetch(self, **kwargs):
        url = "https://civitai.com/api/v1/images"
        urls = []
        kwargs = {k: v for k, v in kwargs.items() if v}
        url = '{}?{}'.format(url, urllib.parse.urlencode(kwargs))
        print(url)
        response = requests.get(url)

        if response.status_code == 200:
            body = response.json()
            print(body.get("metadata"))
            images = [item.get("url") for item in body.get("items")] 
            # print(images)
            urls = images 
            return (urls, body.get("metadata").get("nextCursor"),)
        else:
            print(response.status_code)

        return (urls, "",)



from PIL import Image, ImageOps
import torch
import numpy as np

class KAndyLoadImageFromUrl:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Example.jpg/800px-Example.jpg"
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "kandy"

    def load(self, url):
        # get the image from the url
        headers = {
            "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "accept-language": "en,en-US;q=0.9,ru;q=0.8",
            "cache-control": "no-cache",
            "pragma": "no-cache",
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
   
        image = Image.open(requests.get(url, stream=True, headers=headers).raw)
        image = ImageOps.exif_transpose(image)
        return (self.pil2tensor(image),)

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



import torch
import random

class KAndyNoiseCondition:
    """Adds noise to a conditioning tensor."""

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    CATEGORY = "kandy"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "op_type": (["MUL", "ADD"],),
                "noise_level": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 100.0, "step": 0.01}), # Added noise level input
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffFF})  # Added seed for reproducibility
            }
        }

    def run(self, conditioning, op_type, noise_level, seed):
        if not isinstance(conditioning, list):
            print(f"Warning: Conditioning input is not a list. Got type: {type(conditioning)}")
            return (conditioning,)
    
        noisy_conditioning = []
        for cond in conditioning:
            tensor = cond[0]
            #torch.manual_seed(seed)
            noise = torch.randn_like(tensor, memory_format=torch.contiguous_format) * noise_level # Scale noise by mean absolute value
            if op_type == "MUL":
                noisy_tensor = tensor + tensor * noise
            else:    
                noisy_tensor = tensor + noise * tensor.abs().mean()

            noisy_conditioning.append((noisy_tensor, cond[1]))

        return (noisy_conditioning,)
   

import requests
from bs4 import BeautifulSoup
from lxml import html

class KAndyImagesByCss:
    """Loads images from a URL using CSS selectors or XPath."""

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    
    FUNCTION = "run"
    CATEGORY = "kandy"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING",),
                "selector": ("STRING",),  # Renamed "css" to "selector" for clarity
                "attr": ("STRING", {"default": "href"}),
                "use_xpath": ("BOOLEAN", {"default": False}), # Added option to use XPath
                "limit": ("INT", {"default": 0, "min": 0, "max": 64}),
                "start_from": ("INT", {"default": 0, "min": 0, "max": 64}),
            }
        }

    def run(self, url, selector, attr, use_xpath, limit):
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        html_content = response.text
        #print(html_content)
        if use_xpath:
            tree = html.fromstring(html_content)
            elements = tree.xpath(selector)
            urls = [str(element.get(attr)) for element in elements if element.get(attr)]
        else: #use css
            soup = BeautifulSoup(html_content, "html.parser")
            elements = soup.select(selector)
            urls = [str(img[attr]) for img in elements if img.has_attr(attr)]
        if limit > 0:
            urls = urls[:]
        return (urls,)

    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KAndyCivitPromptAPI": KCivitPromptAPI,
    "KAndyLoadImageFromUrl": KAndyLoadImageFromUrl,
    "KAndyCivitImagesAPI": KCivitImagesAPI,
    "KAndyNoiseCondition": KAndyNoiseCondition,
    "KAndyImagesByCss": KAndyImagesByCss
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KAndyCivitPromptAPI": " Civit Prompt API",
    "KAndyLoadImageFromUrl": "Load Image From Url",
    "KAndyCivitImagesAPI": "Civit Images API",
    "KAndyNoiseCondition": "KAndyNoiseCondition",
    "KAndyImagesByCss": "KAndyImagesByCss"
}

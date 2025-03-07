import requests
import urllib
import importlib
import os


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
                "nsfw": (["Soft", "Mature", "X", "XXX", "All", ""],),
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
                "url": ("STRING",),
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

    def run(self, url, selector, attr, use_xpath, limit, start_from=0):
        urls = []
        # print(f"{url}, {selector}, {attr}, {use_xpath}, {limit}, {start_from}")
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
            urls = [str(img[attr]) for img in elements]
        if start_from > 0:
            urls = urls[start_from:]
        if limit > 0:
            urls = urls[:limit]
        # print(f"{urls}")
        return (urls,)


class KPornImageAPI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT",)
            },
        }
    
    @classmethod
    def IS_CHANGED(c, **kwars):
        return True  

    RETURN_NAMES = ("IMAGES",)
    RETURN_TYPES = ("STRING",)

    FUNCTION = "ffetch"
    CATEGORY = "kandy"

    def ffetch(self, seed):
        url = "https://www.pornpics.com/random/index.php?lang=en"
        print(url)
        response = requests.post(url, json = {'seed': seed})

        if response.status_code == 200:
            body = response.json()
            urls = body.get('link') 
            return (urls,)
        else:
            print(response.status_code)
            return ("",)   

import sqlitedict
import random
import os

class KPromtGen:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT",),
                "n": ("INT",),
                "replace_underscore": ("BOOLEAN", {"default": False})
            },
        }
    
    @classmethod
    def IS_CHANGED(c, **kwars):
        return True 
        

    RETURN_NAMES = ("prompt",)
    RETURN_TYPES = ("STRING",)

    FUNCTION = "gen"
    CATEGORY = "kandy"

    def __init__(self):
        sqlite_db_path = os.path.join(os.path.dirname(__file__), 'data', 'wd-eva02-large-tagger-v3.sqlite')
        self.z_tokens = sqlitedict.SqliteDict(sqlite_db_path)
        self.rng = random.Random()
        pass

    def gen(self, seed, n, replace_underscore = False):
        """ Add n random elements from zTokens array to the prompt """

        components = []
        zLen = len(self.z_tokens)
        for _ in range(n):
            token = self.z_tokens[self.rng.randint(1, zLen - 1)]['name'] 
            token = token.replace("_", " ") if replace_underscore else token
            components.append(token)
        
        self.rng.shuffle(components)
        prompt = ",".join(components)
        return (prompt,)

        
import re

class KandySimplePrompt: 
    def __init__(self):
        self.rnd = random.Random(324532465345)
        self.ii = 42
        pass
     
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}), 
            },
            "optional":{ 
                "add_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "hidden": {"my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT_TXT",)

    FUNCTION = "node"
    CATEGORY = "KAndy"
 
    def node(self, prompt="", add_prompt="", my_unique_id=0): 
        self.ii += 1
        self.rnd.seed(int(my_unique_id) + self.ii)
        out = self.replace_random_part(prompt  + "," + add_prompt)
        return(out,)

    def replace_random_part(self, text): 
        matches = re.findall(r'\{([^{}]*)\}', text)
         
        for match in matches: 
            options = match.split('|')
            replacement = self.rnd.choice(options) 
            text = text.replace('{' + match + '}', replacement, 1)
         
        text = text.replace('|', '').replace('{', '').replace('}', '')
        
        return text


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KAndyCivitPromptAPI": KCivitPromptAPI,
    "KAndyLoadImageFromUrl": KAndyLoadImageFromUrl,
    "KAndyCivitImagesAPI": KCivitImagesAPI,
    "KAndyNoiseCondition": KAndyNoiseCondition,
    "KAndyImagesByCss": KAndyImagesByCss,
    "KPromtGen": KPromtGen,
    "KPornImageAPI": KPornImageAPI,

    "KandySimplePrompt": KandySimplePrompt
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KAndyCivitPromptAPI": " Civit Prompt API",
    "KAndyLoadImageFromUrl": "Load Image From Url",
    "KAndyCivitImagesAPI": "Civit Images API",
    "KAndyNoiseCondition": "NoiseCondition",
    "KAndyImagesByCss": "ImagesByCss",
    "KPromtGen": "Promt Generator",
    "KPornImageAPI": "PornImageAPI",
   
    "KandySimplePrompt": "Simple Prompt"
}


nodes_dir = os.path.join(os.path.dirname(__file__), 'nodes')
node_list = [os.path.splitext(f)[0] for f in os.listdir(nodes_dir) if os.path.isfile(os.path.join(nodes_dir, f)) and f.endswith('.py')]


for module in node_list:
    try:
        imported_module = importlib.import_module(f".nodes.{module}", package="ComfyUI-Kandy")

        module_key = imported_module.__NODE__.__name__
        module_name = ' '.join(word.capitalize() for word in module.split('-'))
        NODE_CLASS_MAPPINGS[module_key] = imported_module.__NODE__
        NODE_DISPLAY_NAME_MAPPINGS[module_key] = module_name
        print(f"[KAndy] Add node: {module_key} {module}")
    except Exception as e:
        print(f"Failed to import module: {module}" + e)
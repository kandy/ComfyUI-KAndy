import requests
import importlib
import os




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
            "accept-language": "en,en-US;q=0.9",
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


# class AnyType(str):
#     """A special type that can be connected to any other types."""
#     def __ne__(self, __value: object) -> bool:
#         return False
# any_type = AnyType("*")


from .tagger import KAndyWD14Tagger
from .tagger import KAndyTaggerModelLoader

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "KAndyLoadImageFromUrl": KAndyLoadImageFromUrl,
    "KPromtGen": KPromtGen,
    "KandySimplePrompt": KandySimplePrompt,
    "KAndyTaggerModelLoader": KAndyTaggerModelLoader,
    "KAndyWD14Tagger": KAndyWD14Tagger,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KAndyLoadImageFromUrl": "Load Image From Url",
    "KAndyCivitImagesAPI": "Civit Images API",
    "KAndyNoiseCondition": "NoiseCondition",
    "KAndyImagesByCss": "ImagesByCss",
    "KPromtGen": "Promt Generator",
    "KPornImageAPI": "PornImageAPI",
   
    "KandySimplePrompt": "Simple Prompt",

    "KAndyTaggerModelLoader": "Tagger ModelLoader",
    "KAndyWD14Tagger": "WD14 Tagger",
    
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
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

        components = ['BREAK']
        zLen = len(self.z_tokens)
        for _ in range(n):
            token = self.z_tokens[self.rng.randint(1, zLen - 1)]['name'] 
            if replace_underscore:
                token = token.replace("_", " ")
            components.append(token)
        
        self.rng.shuffle(components)
        prompt = ",".join(components)
        return (prompt,)

import uuid
from PIL import Image
from datetime import datetime, timezone


class KCivitaiPostAPI:
    """Post a set of images to Civitai"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("STRING", {"is_list": True}),
                "title": ("STRING",),
                "description": ("STRING",),
                "cookie": ("STRING",)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "post_images"
    CATEGORY = "kandy"


    def post_images(self, images, title, description, cookie):
        headers = {
            "accept-language": "en,en-US;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",

            "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "image",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "cross-site",  
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36",

            "accept": "*/*",
            "content-type": "application/json",
            "cookie": cookie,
            "referer": "https://civitai.com/posts/create",
        }

        # Step 1: Create a post
        post_data = {
            "json": {
                "modelVersionId": None,
                "tag": None,
                "authed": True
            },
            "meta": {
                "values": {
                    "modelVersionId": ["undefined"],
                    "tag": ["undefined"]
                }
            }
        }
        # print(f"Request to create post: {post_data}")
        response = requests.post("https://civitai.com/api/trpc/post.create", json=post_data, headers=headers)
        # print(f"Response: {response.status_code}, {response.text}")
        if response.status_code != 200:
            return (f"Failed to create post: {response.status_code}",)
        postJson = response.json().get("result", {}).get("data", {}).get("json", {});
        # print(f"Post created: {postJson}")
        post_id = postJson.get("id")
        if not post_id:
            return (f"Failed to create post: {response.text}",)

        index = 0
        # Step 2: Upload images
        for image_path in images:
            index += 1
            width = 512 
            height = 512
            with Image.open(image_path) as img:
                width, height = img.size
            with open(image_path, "rb") as image_file:
                # Step 2.1: Get upload image url
                image_data = image_file.read()
                # print(f"Request to upload image: {image_path}")
                post_data = {
                    "filename": image_path.split("/")[-1],
                    "type": "image",
                    "size": len(image_data),
                    "mimeType": "image/jpeg"
                }
                upload_response = requests.post("https://civitai.com/api/v1/image-upload/multipart", json=post_data, headers=headers)
                # print(f"Response: {upload_response.status_code}, {upload_response.text}")
                if upload_response.status_code != 200:
                    return (f"Failed to upload image: {upload_response.status_code}",)
                upload_url = upload_response.json().get("urls",{})[0].get("url")
                if not upload_url:
                    return (f"Failed to get upload URL: {upload_response.text}",)    

                # Step 2.2: Put image to s3
                # print(f"Request to put image: {upload_url}")
                put_response = requests.put(upload_url, data=image_data, headers={"content-type": "image/jpeg"})
                # print(f"Response: {put_response.status_code}, {put_response.text}")
                if put_response.status_code != 200:
                    return (f"Failed to put image: {put_response.status_code}",)
                upload_data = upload_response.json()
                post_data = {
                    "bucket": upload_data['bucket'],
                    "key": upload_data['key'],
                    "type": "image",
                    "uploadId": upload_data['uploadId'],
                    "parts": [{"ETag": put_response.headers['ETag'], "PartNumber": 1}],
                }
               

                # Step 2.3: Complete image upload
                complete_response = requests.post("https://civitai.com/api/upload/complete", json=post_data, headers=headers)
                # print(f"Response: {complete_response.status_code}, {complete_response.text}")
                if complete_response.status_code != 200:
                    return (f"Failed to upload image: {complete_response.status_code}",)
                
                hash = 'U9Hd]RJ502wPx]$%i_Iq00sE~AtLIUIqS5-U'
                # TODO: CACLULATE HASH
               
                # Step 3: Add image to post
                image_metadata = {
                    "json": {
                        "name": image_path.split("/")[-1],
                        "url": upload_url.split("?")[0].split("/")[-1],
                        "hash": hash,
                        "height": height,
                        "width": width,
                        "postId": post_id,
                        "modelVersionId": None,
                        "index": index,
                        "mimeType": "image/jpeg",
                        "uuid": str(uuid.uuid4()),
                        "meta": None,
                        "metadata": {
                            "size": len(image_data),
                            "height": height,
                            "width": width,
                            "hash": hash
                        },
                        "externalDetailsUrl": None,
                        "authed": True
                    },
                    "meta": {
                        "values": {
                            "modelVersionId": ["undefined"],
                            "externalDetailsUrl": ["undefined"]
                        }
                    }
                }
                # print(f"Request to add image to post: {image_metadata}")
                add_image_response = requests.post("https://civitai.com/api/trpc/post.addImage", json=image_metadata, headers=headers)
                # print(f"Response: {add_image_response.status_code}, {add_image_response.text}")
                if add_image_response.status_code != 200:
                    return (f"Failed to add image to post: {add_image_response.status_code}",)

        # Step 4: Update post with title and description
        update_data = {
            "json": {
            "id": post_id,
            "title": title,
            "description": description,
            "tags": [],
            "publishedAt": str(datetime.now(timezone.utc).isoformat()),
            "collectionId": None,
            "collectionTagId": None,
            "authed": True
            },
            "meta": {
                "values": {
                    "title": ["undefined"],
                    "description": ["undefined"],
                    "tags": ["undefined"],
                    "publishedAt": ["Date"],
                    "collectionId": ["undefined"]
                }
            }
        }
        # print(f"Request to update post: {update_data}")
        update_response = requests.post("https://civitai.com/api/trpc/post.update", json=update_data, headers=headers)
        # print(f"Response: {update_response.status_code}, {update_response.text}")
        if update_response.status_code != 200:
            return (f"Failed to update post: {update_response.status_code}",)

        return (f"Post created successfully with ID: https://civitai.com/posts/{post_id}",)
        




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
    "KCivitaiPostAPI": KCivitaiPostAPI
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "KAndyCivitPromptAPI": " Civit Prompt API",
    "KAndyLoadImageFromUrl": "Load Image From Url",
    "KAndyCivitImagesAPI": "Civit Images API",
    "KAndyNoiseCondition": "KAndyNoiseCondition",
    "KAndyImagesByCss": "KAndyImagesByCss",
    "KPromtGen": "Promt Generator",
    "KPornImageAPI": "KPornImageAPI",
    "KCivitaiPostAPI": "Civitai Post API"
}

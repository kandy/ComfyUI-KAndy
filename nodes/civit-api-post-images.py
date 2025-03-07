import uuid
import requests
import datetime
from PIL import Image
import piexif

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
            "referer": "https://civitai.com/posts/edit?src=generator",
        }

        # Step 1: Create a post
        post_data = {
            "json": {
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
            prompt = "masterpiece, best quality,  (24yo)"
            with Image.open(image_path) as img:
                width, height = img.size
                # Extract prompt from EXIF metadata
                try:
                    exif_data = img.info.get("exif")
                    if exif_data:
                        exif = piexif.load(exif_data)
                        prompt = exif.get("Exif").get(piexif.ExifIFD.UserComment).decode('utf8').replace('UNICODE\x00', "").replace('\x00', "").split("Negative prompt:")[0]
                except Exception as e:
                    print(f"Failed to read EXIF data: {e}")
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
                        "generationProcess": "txt2img", 
                        "hash": hash,
                        "height": height,
                        "width": width,
                        "postId": post_id,
                        "modelVersionId": None,
                        "index": index,
                        "mimeType": "image/jpeg",
                        "uuid": str(uuid.uuid4()),
                        "meta": {
                            "Size": "832x1216",
                            "nsfw": True,
                            "seed": 1650258278,
                            "draft": False,
                            # "extra": {
                            #     "remixOfId": 54864033
                            # },
                            "steps": 25,
                            "width": 832,
                            "height": 1216,
                            "prompt": prompt,
                            "sampler": "Euler a",
                            "cfgScale": 5,
                            "clipSkip": 2,
                            "quantity": 2,
                            "workflow": "txt2img",
                            "baseModel": "Illustrious",
                            "resources": [],
                            "Created Date": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
                            "negativePrompt": "NSFW,  lowres, worst quality, low quality, bad anatomy, bad hands, 4koma, comic, greyscale, censored, jpeg artifacts, logo, patreon, NUDITY, SUGGESTIVE",
                            "civitaiResources": [
                                {
                                "type": "checkpoint",
                                "modelVersionId": 1429555,
                                "modelVersionName": "v1.0"
                                },
                                {
                                "type": "checkpoint",
                                "modelVersionId": 1564095,
                                "modelVersionName": "v1.0"
                                },
                                {
                                "type": "embed",
                                "weight": 1,
                                "modelVersionId": 250708,
                                "modelVersionName": "safe_pos"
                                },
                                {
                                "type": "embed",
                                "weight": 1,
                                "modelVersionId": 250712,
                                "modelVersionName": "safe_neg"
                                },
                                {
                                "type": "embed",
                                "weight": 1,
                                "modelVersionId": 106916,
                                "modelVersionName": "v1.0"
                                }
                            ]
                        },
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
            "publishedAt": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
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



__NODE__ = KCivitaiPostAPI
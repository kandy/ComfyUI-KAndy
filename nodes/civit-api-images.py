import requests
import urllib


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

__NODE__ = KCivitImagesAPI
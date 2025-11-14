import requests
import json


class KCivitImagesInt:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit": ("INT", {"default": 1, "min": 0,  "step": 1},)
            },
            "optional": { 
                "cursor": ("STRING",),
                "entity": ("STRING",),
                "type": (["All", "Tags","Model"],), 
                "period": (["AllTime", "Year", "Month", "Week", "Day", ""],), 
                "sort": (["Most Reactions", "Most Comments","Newest", ""],),
                "cookies": ("STRING", {"multiline": True},),
            }
        }

    RETURN_NAMES = ("IMAGES", "Cursor",)
    RETURN_TYPES = ("STRING","STRING",)
    OUTPUT_IS_LIST = (True, False,)

    FUNCTION = "ffetch"
    CATEGORY = "KAandy"

    def ffetch(self, limit,  cursor="", entity="", type="Tags", period="AllTime", sort="Most Reactions", cookies=""):
        url = "https://civitai.com/api/trpc/image.getImagesAsPostsInfinite"
        
        urls = []
       
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
            "cookie": cookies,
            "referer": "https://civitai.com/posts/edit?src=generator",
        }




        if cursor == '':
           cursor = None
        print(f"Fetching Civitai Images: type={type}, entity={entity}, period={period}, sort={sort}, cursor={cursor}, limit={limit}")
        data =  {
            "period": period,
            "periodMode": "published",
            "sort": sort,
            "types": [
                "image"
            ],
            "withMeta": False,
            "fromPlatform": False,
            "hideAutoResources": False,
            "hideManualResources": False,
            "notPublished": False,
            "scheduled": False,
            "hidden": False,
            "followed": False,
            "remixesOnly": False,
            "nonRemixesOnly": False,
            "requiringMeta": False,
            "useIndex": True,
            "cursor": cursor,
            "authed": True,
            "browsingLevel": 31,
            "limit": limit,
        }




        if type == "Tags":
            data["tags"] = [int(entity)]
        elif type == "Model":
            url = "https://civitai.com/api/trpc/image.getImagesAsPostsInfinite"
            data["modelId"] = int(entity)
        elif type == "All":
            data = {
                "period": period,
                "periodMode": "published",
                "sort": sort,
                "types": [
                    "image"
                ],
                "withMeta": False,
                "fromPlatform": False,
                "hideAutoResources": False,
                "hideManualResources": False,
                "notPublished": False,
                "scheduled": False,
                "hidden": False,
                "followed": False,
                "remixesOnly": False,
                "nonRemixesOnly": False,
                "requiringMeta": False,
                "useIndex": True,
                "browsingLevel": 31,
                "include": [
                    "cosmetics"
                ],
                "excludedTagIds": [
                    415792,
                    426772,
                    5188,
                    5249,
                    130818,
                    130820,
                    133182,
                    5351,
                    306619,
                    154326,
                    161829,
                    163032
                ],
                "disablePoi": True,
                "disableMinor": True,
                "cursor": "",
                "authed": True
            }
        
        #https://civitai.com/api/trpc/image.getInfinite?input=%7B%22json%22%3A%7B%22period%22%3A%22Day%22%2C%22periodMode%22%3A%22published%22%2C%22sort%22%3A%22Most%20Reactions%22%2C%22types%22%3A%5B%22image%22%5D%2C%22withMeta%22%3Afalse%2C%22fromPlatform%22%3Afalse%2C%22hideAutoResources%22%3Afalse%2C%22hideManualResources%22%3Afalse%2C%22notPublished%22%3Afalse%2C%22scheduled%22%3Afalse%2C%22hidden%22%3Afalse%2C%22followed%22%3Afalse%2C%22remixesOnly%22%3Afalse%2C%22nonRemixesOnly%22%3Afalse%2C%22requiringMeta%22%3Afalse%2C%22useIndex%22%3Atrue%2C%22browsingLevel%22%3A31%2C%22include%22%3A%5B%22cosmetics%22%5D%2C%22excludedTagIds%22%3A%5B415792%2C426772%2C5188%2C5249%2C130818%2C130820%2C133182%2C5351%2C306619%2C154326%2C161829%2C163032%5D%2C%22disablePoi%22%3Atrue%2C%22disableMinor%22%3Atrue%2C%22cursor%22%3Anull%2C%22authed%22%3Atrue%7D%2C%22meta%22%3A%7B%22values%22%3A%7B%22cursor%22%3A%5B%22undefined%22%5D%7D%7D%7D
        input= json.dumps({
            "json": data
        })
        response = requests.get(url, params={"input": input}, headers=headers)
        if response.status_code == 200:
            body = response.json()
            data = body.get("result").get("data").get("json")
            images = [f"https://image.civitai.com/xG1nksdfTMzGDvpLrqFT7Wd/{item.get("url")}/anim=false/{item.get("url","aaaa.jpeg")}" for item in data.get("items")] 
            # print(images)
            urls = images 
            return (urls, data.get("nextCursor"),)
        else:
            print(f"Error: {response.status_code} {response.request.url}\n{response.text}\n{input}\n")

        return (urls, "",)

__NODE__ = KCivitImagesInt


# a = KCivitImagesInt()
# data = a.ffetch(4, "2003939", cursor=104806384, type="Model", period="AllTime", sort="Most Reactions");
# print(data)
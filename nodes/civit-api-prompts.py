import requests
import urllib


class KCivitPromptAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit": (
                    "INT",
                    {"default": 0, "min": 0, "max": 128000, "step": 1},
                )
            },
            "optional": {
                "cursor": ("STRING",),
                "postId": ("STRING",),
                "modelId": ("STRING",),
                "username": ("STRING",),
                "period": (["AllTime", "Year", "Month", "Week", "Day", ""],),
                "nsfw": (["Soft", "Mature", "X", "XXX", "All", ""],),
                "sort": (["Most Reactions", "Most Comments", "Newest", ""],),
            },
        }

    RETURN_NAMES = ("URLs", "cursor")
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )

    CATEGORY = "kandy"

    FUNCTION = "ffetch"

    def ffetch(self, **kwargs):
        url = "https://civitai.com/api/v1/images"
        urls = []
        params = {k: v for k, v in kwargs.items() if v}
        url = "{}?{}".format(url, urllib.parse.urlencode(params))
        print(url)
        response = requests.get(url)

        if response.status_code == 200:
            body = response.json()
            prompts = [
                item.get("meta").get("prompt")
                for item in body.get("items")
                if item.get("meta") and item.get("meta").get("prompt")
            ]
            # print(prompts)
            return (
                prompts,
                body.get("metadata").get("nextCursor"),
            )
        else:
            print(response.status_code)

        return (
            urls,
            "",
        )


__NODE__ = KCivitPromptAPI

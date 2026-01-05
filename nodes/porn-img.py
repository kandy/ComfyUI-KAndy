import requests


class KPornImageAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"seed": ("INT",)},
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
        response = requests.post(url, json={"seed": seed})

        if response.status_code == 200:
            body = response.json()
            urls = body.get("link")
            return (urls,)
        else:
            print(response.status_code)
            return ("",)


__NODE__ = KPornImageAPI

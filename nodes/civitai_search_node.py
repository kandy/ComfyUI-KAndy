import requests
import datetime
import json
from urllib.parse import quote

class CivitaiSearchNode:
    """
    A ComfyUI node to search Civitai for images based on date range and authorization token.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth_token": ("STRING", {"default": "8c46eb2508e21db1e9828a97968d91ab1ca1caa5f70a00e88a2ba1e286603b61"}),
                "start_date": ("STRING", {"default": "2024-01-01", "format": "date"}),
                "end_date": ("STRING", {"default": "2024-12-31", "format": "date"}),
                "limit": ("INT", {"default": 10, "min": 1}),
                "offset": ("INT", {"default": 0, "min": 0}),
                "sort_by": (['stats.tippedAmountCountAllTime:desc', 'stats.reactionCountAllTime:desc', 'createdAtUnix:desc'], {"default": "stats.reactionCountAllTime:desc"}),
                "query": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("urls", "prompt", "success",)
    FUNCTION = "search_civitai"
    CATEGORY = "kandy"
    OUTPUT_IS_LIST = (True, True, False)

    def search_civitai(self, auth_token, start_date, end_date, limit, offset, sort_by = "stats.reactionCountAllTime:desc", query = "", tags = "", model = ""):
        """
        Searches Civitai for images within the specified date range.
        """
        try:
            # Convert dates to Unix timestamps (milliseconds)
            start_date_ts = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_date_ts = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)



            # Construct the API request
            url = "https://search.civitai.com/multi-search"
            headers = {
                "accept": "*/*",
                "accept-language": "en,en-US;q=0.9,ru;q=0.8",
                "authorization": f"Bearer {auth_token}",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "x-meilisearch-client": "Meilisearch instant-meilisearch (v0.13.5) ; Meilisearch JavaScript (v0.34.0)"
            }
            data = {
                "queries": [
                    {
                        "q": query,
                        "indexUid": "images_v6",
                        "attributesToHighlight": [],
                        "limit": limit,
                        "offset": offset,
                        "filter": [
                            f"createdAtUnix>={start_date_ts}",
                            f"createdAtUnix<={end_date_ts}",
                            "type=image",
                        ],
                        "sort": [sort_by],
                    },
                ]
                
            }



            if tags:
                tags_filter = f"\"tagNames\"=\"{tags}\""
                data["queries"][0]["filter"].append(tags_filter)

            if model and int(model.strip()) > 0:
                model = int(model.strip())
                model_filter = f"modelVersionId={model}"
                data["queries"][0]["filter"].append(model_filter)

            print(data)
            response = requests.post(url, headers=headers, json=data)
       
            response.raise_for_status()  # Raise an exception for bad status codes
            print(response.request.url)

            # Parse the response
            json_data = response.json()

            return self.extract(json_data)

        # except requests.exceptions.RequestException as e:
        #     return ([], [], f"Error during API request: {e}",)
        # except json.JSONDecodeError as e:
        #     return ([], [], f"Error decoding JSON response: {e}",)
        # except ValueError as e:
        #     return ([], [], f"Error converting date: {e}",)
        except Exception as e:
            return ([], [], f"An unexpected error occurred: {e}",)

    def extract(self, json_data):
        """
        Extracts image URLs from the API response.
        """
        image_urls = []
        prompts = []
        for result in json_data.get("results", []):
            for hit in result.get("hits", []):
                url = quote(hit.get("url"))
                name = quote(hit.get("name"))
                if url and name:
                    # if name dont have extension, add base on 'mimeType'
                    mime_type = hit.get("mimeType")
                    if name.count(".") == 0 and mime_type:
                        ext = mime_type.split("/")[-1]
                        name += f".{ext}"
                    image_url = f"https://image.civitai.com/xG1nksdfTMzGDvpLrqFT7Wd/{url}/anim=false,original=true,quality=90,optimized=true/{name}"
                    image_urls.append(image_url)
                    prompts.append(hit.get("prompt") or "")
        return (image_urls, prompts, "ok",)

__NODE__ = CivitaiSearchNode

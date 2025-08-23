import requests
import urllib.parse
import json

class DeviantArtAPIImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client_id": ("STRING", {"default": "", "multiline": False}),
                "client_secret": ("STRING", {"default": "", "multiline": False}),
                "date": ("STRING", {"default": "landscape", "multiline": False}),
                "limit": ("INT", {"default": 24, "min": 1, "max": 200, "step": 1}),
                "mature_content": (["true", "false"],),
            },
            "optional": {
                "offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_NAMES = ("IMAGE_URLS", "next",)
    RETURN_TYPES = ("STRING","STRING",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "fetch_images"
    CATEGORY = "kandy/DeviantArt"

    def fetch_images(self, client_id, client_secret, date, limit, mature_content, offset=0):
        # DeviantArt API requires OAuth2 for authentication.
        # First, get an access token.
        token_url = "https://www.deviantart.com/oauth2/token"
        token_params = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        
        try:
            token_response = requests.post(token_url, data=token_params)
            token_response.raise_for_status() # Raise an exception for HTTP errors
            access_token = token_response.json().get("access_token")
            if not access_token:
                print("DeviantArt API Error: Could not retrieve access token.")
                return ([],)
        except requests.exceptions.RequestException as e:
            print(f"DeviantArt API Token Error: {e}")
            return ([],)

        print(f"Access Token: {access_token}")  # Debugging line, remove in production
        # Now, use the access token to fetch images.
        # curl https://www.deviantart.com/api/v1/oauth2/browse/dailydeviations?access_token=Alph4num3r1ct0k3nv4lu3
        api_url = "https://www.deviantart.com/api/v1/oauth2/browse/home" # This is a common pattern, but might not be exact.
        
        headers = {
            "User-Agent": "ComfyUI-DeviantArt-Node/1.0" # Good practice to include a User-Agent
        }

        params = {
            "q": date,
            "limit": limit,
            "offset": offset,
            "mature_content": "true" if mature_content == "true" else "false",
            "access_token": access_token,  # Include the access token in the parameters
        }
        
        # Filter out empty parameters
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        image_urls = []
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()
            print(f"Response Data: {response.text}")  # Debugging line, remove in production
            # Assuming the API returns a list of deviations, and each deviation has a 'content' or 'media' field
            # with a 'src' or 'url' for the image. This part is highly dependent on actual API response structure.
            if "results" in data: # Common for search results
                for item in data["results"]:
                    # This path to the image URL is a placeholder and needs to be adjusted
                    # based on the actual DeviantArt API response structure.
                    if "content" in item and "src" in item["content"]:
                        image_urls.append(item["content"]["src"])
                        
        except requests.exceptions.RequestException as e:
            print(f"DeviantArt API Request Error: {e}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
            return ([],)
        except json.JSONDecodeError as e:
            print(f"DeviantArt API JSON Decode Error: {e}")
            print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
            return ([],)

        return (image_urls,"",)

__NODE__ = DeviantArtAPIImages

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

    
__NODE__ = KAndyImagesByCss
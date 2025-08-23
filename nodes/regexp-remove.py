import re

class TextRegexpRemove:
    """
    A node that removes all occurrences matched by a regular expression from the input text.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "regexp": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remove_matches"
    CATEGORY = "KAndy/Text"

    def remove_matches(self, text, regexp):
        try:
            cleaned_text = re.sub(regexp, '', text)
            return (cleaned_text,)
        except re.error as e:
            print(f"Error in TextRegexpRemove node: Invalid regex - {e}")
            # Return original text if regex is invalid
            return (text,)

__NODE__ = TextRegexpRemove

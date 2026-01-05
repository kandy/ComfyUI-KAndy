import random


class KAndyShuffleTags:
    """
    A custom node for shuffling tags.

    This node takes a string of comma-separated tags as input, shuffles the tags randomly,
    and returns the shuffled tags as a single comma-separated string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"multiline": True}),
            }
        }

    @classmethod
    def IS_CHANGED(c, **kwars):
        return True

    RETURN_TYPES = ("STRING",)
    FUNCTION = "shuffle_tags"
    CATEGORY = "KAndy"

    def shuffle_tags(self, tags):
        tag_list = tags.split(",")
        random.shuffle(tag_list)
        return (",".join(tag_list),)


__NODE__ = KAndyShuffleTags

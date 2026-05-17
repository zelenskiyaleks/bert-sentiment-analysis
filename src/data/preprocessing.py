import re

def remove_html_tags(text):
    """
    Remove HTML tags from text.
    """
    return re.sub(r"<br\s*/?>", " ", text)

def remove_extra_spaces(text):
    """
    Normalize whitespace in text.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_text(text):
    """
    Apply full text cleaning pipeline.
    """
    text = remove_html_tags(text)
    text = remove_extra_spaces(text)

    return text
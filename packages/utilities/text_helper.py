import html

from th_preprocessor.preprocess import (
    normalize_accented_chars,
    normalize_special_chars,
    remove_tag,
    normalize_link,
    normalize_at_mention,
    normalize_email,
    normalize_phone,
    normalize_haha,
    remove_hashtags,
    replace_dup_chars,
    replace_dup_emojis,
    replace_text,
    normalize_emoji,
    remove_dup_spaces,
    normalize_filename,
)
from th_preprocessor.data import THAI_NORMALIZE_PAIRS, THAI_TO_ARABIC_DIGIT_PAIRS
COMBINED_NORMALIZE_PAIRS = THAI_NORMALIZE_PAIRS + THAI_TO_ARABIC_DIGIT_PAIRS

def text_preprocessor(text: str) -> str:
    text = text.lower()
    text = normalize_accented_chars(text)
    text = normalize_special_chars(text)
    text = html.unescape(text)
    text = remove_tag(text)
    text = normalize_at_mention(text)
    text = normalize_email(text)
    text = normalize_link(text)
    text = normalize_filename(text)
    text = normalize_phone(text)
    text = normalize_haha(text)
    text = remove_hashtags(text)
    text = replace_dup_chars(text)
    text = replace_dup_emojis(text)
    text = replace_text(text, COMBINED_NORMALIZE_PAIRS)
    text = normalize_emoji(text)
    text = remove_dup_spaces(text)

    return text
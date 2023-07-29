from loguru import logger
from pydash import deburr, slugify, truncate
from deforum.data import filler_words
from nltk import word_tokenize
import nltk


def normalize_text(prompt, sep="-", truncate_to=64):
    words = deburr(prompt)
    words = slugify(words, separator=" ")
    # words = words.split(sep)
    try:
        words = [word for word in word_tokenize(words) if word not in filler_words]
    except Exception as e:
        logger.debug(f"Failed to tokenize words! Will attempt to download nltk punkt tokenizer and try again...")
        nltk.download("punkt")
        words = [word for word in word_tokenize(words) if word not in filler_words]
    words = slugify(words, separator=sep)
    words = truncate(words, length=truncate_to, omission="").strip(sep)
    return words

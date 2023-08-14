try:
    import re2 as re
except ImportError:
    import re

from pathlib import Path
from string import Template
from typing import Dict, Union

import nltk
from loguru import logger
from nltk import word_tokenize
from pydash import deburr, slugify, truncate

from deforum.data import filler_words


class TemplateParser(Template):
    """A string class for supporting $-substitutions."""

    delimiter = "$"
    # See https://bugs.python.org/issue31672
    idpattern = r"(?a:[a-z][a-z0-9]*)"
    braceidpattern = None
    flags = re.IGNORECASE


def normalize_text(prompt, sep="-", truncate_to=64):
    """
    Process a prompt into a normalized string without punctuation, stopwords, filler words, or special characters for use in filenames.
    """

    words = deburr(prompt)
    words = slugify(words, separator=" ")

    try:
        words = [word for word in word_tokenize(words) if word not in filler_words]
    except Exception as e:
        logger.debug(f"Failed to tokenize words! Will attempt to download nltk punkt tokenizer and try again...")
        nltk.download("punkt")
        words = [word for word in word_tokenize(words) if word not in filler_words]
    words = slugify(words, separator=sep)
    words = truncate(words, length=truncate_to, omission="").strip(sep)
    return words


def buffer_index_to_digits(index: int, digits: int):
    """
    Convert an integer index to a string with a fixed number of digits padded with zeros.
    """
    index = str(index)
    if len(index) < digits:
        index = "0" * (digits - len(index)) + index
    return index


def find_next_index_in_template(
    template_index_key: str,
    template_string: TemplateParser,
    kwargs: Dict[str, Union[int, str, bool, float]],
    minimum_index: int = 0,
):
    """
    Find the next available index in a template string via globbing the parent directory.
    """

    actual_idx = kwargs.get(template_index_key, minimum_index)
    kwargs[template_index_key] = "*"
    search_str = template_string.safe_substitute(**kwargs)
    parent = Path(search_str).parent.resolve().absolute().as_posix()
    globstr = Path(search_str).name
    paths = list(Path(parent).glob(globstr))
    prefix, suffix = globstr.split("*")
    if len(paths) == 0:
        return actual_idx
    else:
        return max([int(path.stem.replace(prefix, "").replace(suffix, "")) for path in paths]) + 1

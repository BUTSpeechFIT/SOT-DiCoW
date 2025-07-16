"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""
import json
import os
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizerNSF


def get_text_norm(t_norm: str):
    if t_norm == 'whisper':
        import os
        _spelling_file = os.path.join(os.path.dirname(__file__), "english.json")
        SPELLING_CORRECTIONS = json.load(open(_spelling_file))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    else:
        return lambda x: x

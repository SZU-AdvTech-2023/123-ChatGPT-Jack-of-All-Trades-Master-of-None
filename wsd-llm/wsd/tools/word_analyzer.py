#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""
spacy
pip install -U spacy
python -m spacy download en_core_web_trf

Email: liyuncong@idea.edu.cn
Author: Yuncong Li
"""

from typing import List
import json
import os

import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from spacy.tokens import Doc

from wsd.utils import log_utils
from wsd.common import common_path
from wsd.data import wsd_data
from wsd.model.chatgpt import wsd_chatgpt

logger = log_utils.get_logger(__file__)


class WordAnalyzer:
    def get_word_info(self, words: List[str], target_word_index: int = 0):
        raise NotImplementedError


class SpacyWordAnalyzer(WordAnalyzer):
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """

        :param model_name: en_core_web_trf, en_core_web_sm
        """
        # https://spacy.io/models
        # https://spacy.io/models/en#en_core_web_trf
        self.nlp = spacy.load(model_name,
                              exclude=['parser', 'ner']
                              )

    def word_tokenize(self, text: str):
        """

        :param text:
        :return:
        """
        doc = self.nlp(text)
        result = [e.text for e in doc]
        return result

    def word_tokenize_for_wsd(self, word: str, pre_context: str, post_context: str):
        """

        :param word_info:
        :return:
        """
        if pre_context:
            pre_words = self.word_tokenize(pre_context)
        else:
            pre_words = []

        if post_context:
            post_words = self.word_tokenize(post_context)
        else:
            post_words = []
        words = pre_words + [word] + post_words
        target_word_index = len(pre_words)
        result = {
            'words': words,
            'target_word_index': target_word_index
        }
        return result

    def get_word_info(self, words: List[str], target_word_index: int = 0):
        """

        :param words:
        :param target_word_index:
        :return:
        """
        # https://spacy.io/api/doc
        # doc = self.nlp(' '.join(words))
        spaces = [True for _ in words]
        spaces[-1] = False
        raw_doc = Doc(self.nlp.vocab, words=words, spaces=spaces)
        doc = self.nlp(raw_doc)
        target_word = doc[target_word_index]
        result = {
            'pos': target_word.pos_,
            'lemma': target_word.lemma_,
            'word': words[target_word_index],
            'words': words,
            'target_word_index': target_word_index
        }
        return result


class WordNetWordAnalyzer(WordAnalyzer):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def get_word_info(self, words: List[str], target_word_index: int = 0):
        """

        :param words:
        :param target_word_index:
        :return:
        """
        lemma = self.lemmatizer.lemmatize(words[target_word_index])
        pos = nltk.tag.pos_tag(words, tagset='universal')[target_word_index][1]
        result = {
            'lemma': lemma,
            'pos': pos,
            'word': words[target_word_index],
            'words': words,
            'target_word_index': target_word_index
        }
        return result


if __name__ == '__main__':
    # dictionary = Dictionary()
    # word = 'glosses'
    # pre_context = 'Difficult expressions are explained in the'
    # post_context = 'at the bottom of the page.'
    # glosses = dictionary.disambiguate(word, pre_context, post_context)
    # print(json.dumps(glosses, ensure_ascii=False, indent=4))

    word_analyzer = SpacyWordAnalyzer()
    result = word_analyzer.get_word_info(['Directions'], target_word_index=0)
    print(result)

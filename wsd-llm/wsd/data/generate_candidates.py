#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""

生成
datasets/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt
Email: liyuncong@idea.edu.cn
Author: Yuncong Li
"""

import os

from nltk.corpus import wordnet as wn

from wsd.common import common_path
from wsd.tools import nltk_wordnet


def generate_key(lemma, pos):
    key = '{}+{}'.format(lemma, pos)
    return key


def load_wn_senses(wn_senses_path: str):
    wn_senses = {}
    with open(wn_senses_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip().split('\t')
            lemma = line[0]
            pos = line[1]
            senses = line[2:]
            key = generate_key(lemma, pos)
            wn_senses[key] = senses
    return wn_senses


def is_the_same(senses: dict, new_senses: dict):
    """

    :param senses:
    :param new_senses:
    :return:
    """
    keys = sorted(list(senses.keys()))
    keys_str = '##'.join(keys)
    new_keys = sorted(list(new_senses.keys()))
    new_keys_str = '##'.join(new_keys)
    if keys_str == new_keys_str:
        return True
    else:
        return False


if __name__ == '__main__':
    candidate_filepath = os.path.join(common_path.project_dir,
                                      'data/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt')
    wn_senses = load_wn_senses(candidate_filepath)
    errno = 0
    for key in wn_senses.keys():
        word, pos = key.split('+')
        new_senses = nltk_wordnet.query_word_senses(word, pos)

        sensekey_arr = wn_senses[key]
        if key.startswith('door+'):
            print()
        senses = {}
        for s in sensekey_arr:
            definition = wn.lemma_from_key(s).synset().definition()
            senses[s] = definition
        if not is_the_same(senses, new_senses):
            errno += 1
            is_the_same(senses, new_senses)
            print(f'{errno}: {key}')
    print('end')

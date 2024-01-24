#!/usr/bin/env python
# -*- coding:utf-8 -*-　
"""

Email: liyuncong@idea.edu.cn
Author: Yuncong Li
"""

import os
import argparse
import json

from tqdm import tqdm
from nltk import word_tokenize

from wsd.data import wsd_data
from wsd.common import common_path
from wsd.evaluate import evaluate
from wsd.model.chatgpt import chatgpt
from wsd.utils import file_utils


def load_prediction(filepath: str):
    """

    :param filepath:
    :return:
    """
    result = []
    if os.path.exists(filepath):
        result = file_utils.read_all_lines(filepath)
    return result


def main(dataset_name: str, model: str = 'gpt-4-0613', dict_name: str = 'wordnet'):
    """
    models:
    gpt-3.5-turbo-0301
    gpt-3.5-turbo-0613
    gpt-4
    gpt-4-0613
    :param dataset_name:
    :param model:
    :param dict_name:
    :return:
    """
    # load data
    input_base_dir = os.path.join(common_path.project_dir, 'datasets/wsd-hard-benchmark/wsd_hard_benchmark/')
    data_filepath = os.path.join(input_base_dir, dataset_name, '%s.data.xml' % dataset_name)
    key_filepath = os.path.join(input_base_dir, dataset_name, '%s.gold.key.txt' % dataset_name)
    docs = wsd_data.Documents(data_filepath, key_filepath,
                              common_path.wsd_evaluation_framework,
                              extra_context=False, add_special_token=True)

    # inference
    true = docs.keys
    instances = docs.get_all_instances()

    output_base_dir = os.path.join(common_path.project_dir,
                                   'datasets/wsd-hard-benchmark/evaluation/predictions')
    pred_filepath = os.path.join(output_base_dir,
                                 dataset_name,
                                 '%s-predictions.%s.key.txt' % (model, dataset_name))
    pred = load_prediction(pred_filepath)
    output = {
        'current_word_index': 0,
        'data': {}
    }
    data = output['data']
    serial_num = 0
    pos_mapping = {
        'a': 'ADJ',
        'n': 'NOUN',
        'r': 'ADV',
        'v': 'VERB',
        's': 'ADJ_SAT'
    }
    for i, pred_instance in enumerate(pred):
        key, answer = pred_instance.split(' ')
        true_answer = true[key]
        if answer in true_answer:
            continue
        instance = instances[i]
        pos = instance.pos
        lemma = instance.lemma

        delimeter = '<target>' + instance.word + '</target>'
        parts = instance.context.split(delimeter)
        left = parts[0]
        right = parts[1]
        words_left = word_tokenize(left) if left else []
        words_right = word_tokenize(right) if right else []
        if not left:
            k = 0
        else:
            k = len(words_left)
        words = words_left + [instance.word] + words_right

        word_info = {
            'paragraph_id': -1,
            'sentence_id': -1,
            'orginal_word_index': k,
            'context': words,
            'pos': pos,
            'lemma': lemma,
            'serial_num': serial_num,
            'sense_id_candidate': answer,
            'sense_id': true_answer[0]
        }
        data[serial_num] = word_info
        serial_num += 1

    output_filepath = pred_filepath + '.error.json'
    with open(output_filepath, mode='w', encoding='utf-8') as output_file:
        json.dump(output, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate storyline generation')
    parser.add_argument('--dataset_name', type=str, required=False, default='hardEN')
    args = parser.parse_args()
    main(args.dataset_name)


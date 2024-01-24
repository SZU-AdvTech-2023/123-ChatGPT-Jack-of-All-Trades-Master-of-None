import os
import json

from wsd.common import common_path
from wsd.data import wsd_data
from wsd.model.chatgpt import inference
from wsd.utils import file_utils


def load_cases(filepath):
    """

    """
    lines = file_utils.read_all_lines(filepath)
    result = {line.split('\t')[0]: line.split('\t')[1:] for line in lines}
    return result


if __name__ == '__main__':
    filepath1 = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Output_Systems_ALL/gpt-3.5-turbo-0301.key.erroneous_cases')
    filepath2 = os.path.join(common_path.project_dir, 'scorer.log')

    cases1 = load_cases(filepath1)
    cases2 = load_cases(filepath2)

    keys1: set = set(cases1.keys())
    keys2: set = set(cases2.keys())
    intersection = keys1.intersection(keys2)
    diff = dict()
    for key in keys1.union(keys2):
        if key in intersection:
            continue
        if key in keys1:
            des = 'keys1_only'
            labels = '\t'.join(cases1[key])
        else:
            des = 'keys2_only'
            labels = '\t'.join(cases2[key])
        diff[key] = {
            'key': key,
            'des': des,
            'lab': labels
        }

    file_utils.write_lines([json.dumps(diff)], filepath1 + '.compare.json')
    print('')
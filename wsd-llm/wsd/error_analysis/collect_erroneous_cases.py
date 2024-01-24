import os

from wsd.common import common_path
from wsd.data import wsd_data
from wsd.model.chatgpt import inference
from wsd.utils import file_utils


def evaluate(pred, true):
    """

    :param pred:
    :param true:
    :return:
    """
    correct_num = 0
    true_num = 0
    pred_num = len(pred)
    for key, prediction in pred.items():
        label = true[key]
        true_num += len(label)
        if prediction in label:
            correct_num += 1
    precision = float(correct_num) / pred_num
    recall = float(correct_num) / true_num
    f1 = 2 * precision * recall / (precision + recall)
    result = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return result


if __name__ == '__main__':
    data_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                          'ALL.data.xml')
    key_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                         'ALL.gold.key.txt')
    docs = wsd_data.Documents(data_filepath, key_filepath, common_path.wsd_evaluation_framework)

    # inference
    true = docs.keys
    instances = docs.get_all_instances()

    model: str = 'gpt-3.5-turbo-0301'
    predictions = {}
    output_filepath = os.path.join(common_path.project_dir,
                                   'datasets/WSD_Evaluation_Framework/Output_Systems_ALL',
                                   '%s.key' % model)
    output_lines = inference.load_prediction(output_filepath)
    for line in output_lines:
        parts = line.split(' ')
        if len(parts) == 2:
            predictions[parts[0]] = parts[1]
        else:
            predictions[parts[0]] = 'None'

    erroneous_cases = []
    for key, value in predictions.items():
        if 'semeval2015.d000.s016.t008' in key:
            print()
        label = true[key]
        skip = False
        for e in value.split('##'):
            if e in label:
                skip = True
                break
        if skip:
            continue
        line = '%s\t%s\t%s' % (key, value, '\t'.join(label))
        erroneous_cases.append(line)
    file_utils.write_lines(erroneous_cases, output_filepath + '.erroneous_cases')


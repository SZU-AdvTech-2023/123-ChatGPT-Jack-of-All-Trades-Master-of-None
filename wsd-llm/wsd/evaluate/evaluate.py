import os
import subprocess

from wsd.common import common_path
from wsd.data import wsd_data
from wsd.model.chatgpt import inference


def evaluate(pred, true):
    """

    :param pred:
    :param true:
    :return:
    """
    correct_num = 0
    true_num = len(true)
    # true_num = 30
    pred_num = len(pred)
    # todo: 预测多个 (datasets/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java)
    for key, prediction in pred.items():
        label = [e.strip() for e in true[key]]
        separated_prediction = prediction.split('##')
        for e in separated_prediction:
            if e.strip() in label:
                correct_num += 1
                break
    precision = float(correct_num) / pred_num
    recall = float(correct_num) / pred_num
    f1 = 2 * precision * recall / (precision + recall)
    result = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return result


def evaluate_output(scorer_path, gold_filepath, out_filepath):
    eval_cmd = ['java', scorer_path, gold_filepath, out_filepath]
    output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
    output = [x.decode("utf-8") for x in output.splitlines()]
    p, r, f1 = [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


if __name__ == '__main__':
    data_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                          'ALL.data.xml')
    key_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                         'ALL.gold.key.txt')
    docs = wsd_data.Documents(data_filepath, key_filepath, common_path.wsd_evaluation_framework)

    # inference
    true = docs.keys
    instances = docs.get_all_instances()

    model: str = 'gpt-3.5-turbo-0301.context_3_sentence'
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
            predictions[parts[0]] = ''
    metrics = evaluate(predictions, true)
    # metrics_temp = evaluate_output(common_path.scorer_path, key_filepath, output_filepath)
    print(metrics)

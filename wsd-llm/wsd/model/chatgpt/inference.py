import sys
sys.path.append("E:\\ProgramCode\\02_PY_PRO\\local_projects\\wsd\\wsd-models-llm")
import os
from typing import List
from collections import defaultdict

from tqdm import tqdm

from wsd.data import wsd_data
from wsd.common import common_path
from wsd.evaluate import evaluate
from wsd.model.chatgpt import chatgpt
from wsd.utils import file_utils
from wsd.model.chatgpt import wsd_chatgpt
from wsd.wsd_hard_benchmark.evaluation import evaluate_macro_F1, evaluate_micro_F1

def load_prediction(filepath: str):
    """

    :param filepath:
    :return:
    """
    result = []
    if os.path.exists(filepath):
        result = file_utils.read_all_lines(filepath)
    return result

def instance_to_word_info(instance: wsd_data.Word):
    """

    :param instance:
    :return:
    """
    pre_context_end = instance.context.find('<target>')
    pre_context = instance.context[: pre_context_end]

    post_context_start = instance.context.find('</target>') + len('</target>')
    post_context = instance.context[post_context_start:]
    word = instance.word

    result = {
        'word': word,
        'pre_context': pre_context,
        'post_context': post_context
    }
    return result

def senses_to_glosses(senses: dict):
    """

    :param senses:
    :return:
    """
    result = []
    for key, sense in senses.items():
        gloss = {
            'sense_id': key,
            'sense': {
                'first': sense
            }
        }
        result.append(gloss)
    return result

def analyze_distribution(all_senses: List[List[str]]):
    """

    :param all_senses:
    :return:
    """
    all_sense_num = [len(e) for e in all_senses]
    total_sense = sum(all_sense_num)
    print(f'total sense: {total_sense}')

    sense_num_distribution = defaultdict(int)
    for e in all_sense_num:
        sense_num_distribution[e] += 1
    sense_num_distribution = sorted(sense_num_distribution.items(), key=lambda x: x[0])
    print(sense_num_distribution)

def analyze_instances(instances: List[wsd_data.Word]):
    """

    :param instances:
    :return:
    """
    all_senses = [e.senses for e in instances]
    analyze_distribution(all_senses)

def main():
    # load data
    # input_base_dir = os.path.join(common_path.project_dir, 'data/wsd-hard-benchmark/wsd_hard_benchmark/')
    # input_base_dir = os.path.join(common_path.project_dir, 'data\\wsd-hard-benchmark\\wsd_hard_benchmark\\')
    # dataset_name = '42D'
    input_base_dir = os.path.join(common_path.project_dir, 'data\\WSD_Evaluation_Framework\\Evaluation_Datasets\\')
    dataset_name = 'ALL'
    data_filepath = os.path.join(input_base_dir, dataset_name, '%s.data.xml' % dataset_name)
    key_filepath = os.path.join(input_base_dir, dataset_name, '%s.gold.key.txt' % dataset_name)
    docs = wsd_data.Documents(data_filepath, key_filepath,
                              common_path.wsd_evaluation_framework,
                              add_special_token=True)

    # inference
    true = docs.keys
    instances = docs.get_all_instances()
    analyze_instances(instances)

    is_test = False     # test all
    # is_test = True
    if is_test:
        instances_num = 30
    else:
        instances_num = len(instances)

    # models:
    # gpt-3.5-turbo-0301
    # gpt-3.5-turbo-0613
    # gpt-4
    # gpt-4-0613
    # model: str = 'gpt-3.5-turbo-0613'
    model: str = 'gpt-3.5-turbo-1106'
    version = ''
    wsd_model = wsd_chatgpt.WsdChatGPT(model=model, max_try_num=10)
    # output_base_dir = os.path.join(common_path.project_dir,
    #                                'data/wsd-hard-benchmark/evaluation/predictions')
    output_base_dir = os.path.join(common_path.project_dir,
                                   'data\\wsd-hard-benchmark\\evaluation\\predictions')
    # output_filepath = os.path.join(output_base_dir,
    #                                dataset_name,
    #                                '%s%s-predictions.%s.key.txt' % (model, version, dataset_name))
    output_filepath = os.path.join(output_base_dir,
                                   dataset_name,
                                   '%s%s-predictions.%s.key.txt' % (model, version, dataset_name))

    output_lines = load_prediction(output_filepath)

    for i in tqdm(range(instances_num)):
        if i < len(output_lines):
            continue

        instance = instances[i]
        # prediction = chatgpt.predict(instance, model=model)
        word_info = instance_to_word_info(instance)

        senses = instance.senses
        glosses = senses_to_glosses(senses)

        prediction = wsd_model.predict(word_info, glosses, sense_key_prefix='')
        print("\nprediction: " + str(prediction))
        if not prediction:
            prediction = ['error']

        line = '%s %s' % (instance.instance_id, ' '.join(prediction))
        output_lines.append(line)

        if len(output_lines) > 0 and len(output_lines) % 10 == 0:
            file_utils.write_lines(output_lines, output_filepath)

    file_utils.write_lines(output_lines, output_filepath)
    analyze_distribution([e.split(' ')[1:] for e in output_lines])

    correct_instances = []
    erroneous_instances = []

    for i, instance in enumerate(instances[:instances_num]):
        i_pred = output_lines[i].split(' ')[1:]
        i_true = true[instance.instance_id]
        i_correct = [e for e in i_pred if e in i_true]
        if i_correct:
            correct_instances.append(instance)
        else:
            erroneous_instances.append(instance)
    analyze_instances(correct_instances)
    analyze_instances(erroneous_instances)

    # evaluate
    predictions = {line.split(' ')[0]: '##'.join(line.split(' ')[1:]) for line in output_lines}
    metrics = evaluate.evaluate(predictions, true)
    print(metrics)

    # evaluate M-f1 & m-fi
    gold_keys = evaluate_macro_F1.load_keys(key_filepath)
    pred_keys = evaluate_macro_F1.load_keys(output_filepath)

    if is_test:
        gold_keys = {k: gold_keys[k] for k in list(gold_keys.keys())[:instances_num]}

    evaluate_macro_F1.evaluate(gold_keys, pred_keys)
    evaluate_micro_F1.evaluate(gold_keys, pred_keys)

if __name__ == '__main__':
    main()
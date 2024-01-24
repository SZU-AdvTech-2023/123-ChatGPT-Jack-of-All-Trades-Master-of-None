import os

from tqdm import tqdm

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
    lines = file_utils.read_all_lines(filepath)
    return lines


def main():
    # load data
    data_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                          'ALL.data.xml')
    key_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                         'ALL.gold.key.txt')
    docs = wsd_data.Documents(data_filepath, key_filepath)

    # inference
    true = docs.keys
    instances = docs.get_all_instances()

    # models:
    # gpt-3.5-turbo-0301
    # gpt-3.5-turbo-0613
    # gpt-4
    model: str = 'gpt-4'
    predictions = {}
    case_id = 'senseval2.d000.s003.t009'
    for i in tqdm(range(len(instances))):
        instance = instances[i]
        if instance.instance_id != case_id:
            continue

        prediction = chatgpt.predict(instance, model=model, debug=True)
        predictions[instance.instance_id] = prediction
    print(predictions)


if __name__ == '__main__':
    main()

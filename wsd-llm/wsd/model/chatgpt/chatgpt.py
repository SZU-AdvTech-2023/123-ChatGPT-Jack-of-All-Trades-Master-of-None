from typing import List
import traceback
import time

from wsd.data import wsd_data
from wsd.utils import chatgpt_client

CHATGPT_URL = "http://198.11.177.165:10001/api/chat_gpt_api"


def answer_engineer(keys: List[str], answer: str):
    """

    :param keys:
    :param answer:
    :return:
    """
    for key in keys:
        if key in answer:
            return key
    return answer


def predict(instance: wsd_data.Word, model: str = 'gpt-3.5-turbo-0301', debug=False, max_tokens=100):
    """

    :param instance:
    :param model:
    :param debug:
    :param max_tokens:
    :return:
    """
    sentence = instance.context
    word = instance.word
    lemma = instance.lemma
    temp = instance.senses
    if len(temp) == 0:
        return '', ''
    if len(temp) == 1:
        answer = list(temp.keys())[0]
        value = temp[answer]
        return [(answer, value)]

    keys = {}
    for key, value in temp.items():
        keys[key] = value
    senses = str(temp)
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are a english linguist."
        # },
        {
            "role": "user",
            "content": """Which meaning of the word „{word}” is expressed in the following context: "{sentence}" The meanings are as follows: {senses}. 
            Return only the key of the most relevant meaning.
                """
            .format(word=word, sentence=sentence, senses=senses)
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        'max_tokens': max_tokens
    }

    if debug:
        print(messages[0]['content'])

    answer = ''
    max_try_num = 10
    try_num = 0
    while try_num < max_try_num:
        try_num += 1
        # res = requests.post(url=CHATGPT_URL, data=data).text
        # res_obj = json.loads(res)
        try:
            res_obj = chatgpt_client.chat(**data)
            if res_obj['choices'][0]['finish_reason'] == 'stop':
                answer = res_obj['choices'][0]['message']['content']
                break
        except:
            traceback.print_exc()
            time.sleep(3)
    answer = answer_engineer(keys, answer)
    if answer in keys:
        value = keys[answer]
    else:
        value = ''

    # phonetic_symbol = Pair()
    # pos = Pos()
    # simple_sense = ''
    # sense_id: int = 0
    # sense: Pair = Pair(first=value)
    # examples: List[Pair] = []
    # gloss = Gloss(phonetic_symbol, pos, simple_sense, sense_id, sense, examples)
    return [(answer, value)]


def predict_word_more_than_one_time(instance: wsd_data.Word, model: str = 'gpt-3.5-turbo-0301', debug=False, max_tokens=100):
    """

    :param instance:
    :param model:
    :param debug:
    :param max_tokens:
    :return:
    """
    sentence = instance.context
    word = instance.word
    lemma = instance.lemma
    temp = instance.senses
    if len(temp) == 0:
        return '', ''
    if len(temp) == 1:
        answer = list(temp.keys())[0]
        value = temp[answer]
        return [(answer, value)]

    keys = {}
    for key, value in temp.items():
        keys[key] = value
    senses = str(temp)
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are a english linguist."
        # },
        {
            "role": "user",
            "content": """Which meaning of the word „{word}” between <target> and </target> in the following context is expressed: "{sentence}" The meanings are as follows: {senses}. 
            Return only the key of the most relevant meaning.
                """
            .format(word=word, sentence=sentence, senses=senses)
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        'max_tokens': max_tokens
    }

    if debug:
        print(messages[0]['content'])

    answer = ''
    max_try_num = 10
    try_num = 0
    while try_num < max_try_num:
        try_num += 1
        # res = requests.post(url=CHATGPT_URL, data=data).text
        # res_obj = json.loads(res)
        try:
            res_obj = chatgpt_client.chat(**data)
            if res_obj['choices'][0]['finish_reason'] == 'stop':
                answer = res_obj['choices'][0]['message']['content']
                break
        except:
            traceback.print_exc()
            time.sleep(3)
    answer = answer_engineer(keys, answer)
    if answer in keys:
        value = keys[answer]
    else:
        value = ''

    # phonetic_symbol = Pair()
    # pos = Pos()
    # simple_sense = ''
    # sense_id: int = 0
    # sense: Pair = Pair(first=value)
    # examples: List[Pair] = []
    # gloss = Gloss(phonetic_symbol, pos, simple_sense, sense_id, sense, examples)
    return [(answer, value)]


def predict_topn(instance: wsd_data.Word, model: str = 'gpt-3.5-turbo-0301', debug=False, max_tokens=100):
    """

    :param instance:
    :param model:
    :param debug:
    :param max_tokens:
    :return:
    """
    sentence = instance.context
    word = instance.word
    lemma = instance.lemma
    temp = instance.senses
    if len(temp) == 0:
        return '', ''
    if len(temp) == 1:
        answer = list(temp.keys())[0]
        value = temp[answer]
        return [(answer, value)]

    keys = {}
    for key, value in temp.items():
        keys[key] = value
    senses = str(temp)
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are a english linguist."
        # },
        {
            "role": "user",
            "content": """Which meaning of the word „{word}” is expressed in the following context: "{sentence}" The meanings are as follows: {senses}. 
            Return only the keys of the three most relevant meaning and arrange them in descending order of relevance.
                """.format(word=word, sentence=sentence, senses=senses)
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        'max_tokens': max_tokens
    }

    if debug:
        print(messages[0]['content'])

    answer = ''
    max_try_num = 10
    try_num = 0
    while try_num < max_try_num:
        try_num += 1
        # res = requests.post(url=CHATGPT_URL, data=data).text
        # res_obj = json.loads(res)
        try:
            res_obj = chatgpt_client.chat(**data)
            if res_obj['choices'][0]['finish_reason'] == 'stop':
                answer = res_obj['choices'][0]['message']['content']
                break
        except:
            traceback.print_exc()
            time.sleep(3)
    return [(answer, '')]


if __name__ == '__main__':
    sentence = 'Marble can be polished to a high <target>gloss</target>. lip gloss.'
    original_word = 'gloss'
    lemma = 'gloss'
    # temp = nltk_wordnet.query_word_senses(original_word)
    word_glosses = load_parsed_glosses.load_glosses()
    res = []
    if lemma in word_glosses:
        temp = {}
        temp_gloss_obj = {}
        for gloss in word_glosses[lemma]:
            temp[str(gloss['sense_id'])] = gloss['sense']['first']
            temp_gloss_obj[str(gloss['sense_id'])] = gloss
        word = wsd_data.Word(original_word, lemma, '', sentence, senses=temp)

        # res_temp = predict(word, debug=True)
        res_temp = predict_word_more_than_one_time(word, model='gpt-3.5-turbo-0301')
        for gloss_candidate in res_temp:
            if gloss_candidate[1]:
                res.append(temp_gloss_obj[gloss_candidate[0]])
        print(res)

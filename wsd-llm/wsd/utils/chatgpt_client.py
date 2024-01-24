"""
https://openai.com/blog/chatgpt
https://platform.openai.com/docs/guides/chat
"""
import time
import argparse
import json
import traceback

import openai

openai.api_base = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
openai.api_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxx'


def chat(**kwargs):
    """

    :param messages:
    :param model_name:
    :param request_timeout:
    :param length
    :return:
    """
    # return messages
    response = openai.ChatCompletion.create(**kwargs)
    return response


def simple_chat(messages, model: str = 'gpt-3.5-turbo-0613', max_tokens: int = 200, request_timeout: int = 10,
                max_try_num: int = 3):
    """

    :param messages:
    :param model:
    :param max_tokens:
    :param max_try_num:
    :param request_timeout:
    :return:
    """
    data = {
        "model": model,
        "messages": messages,
        'max_tokens': max_tokens,
        'request_timeout': request_timeout,
        # "temperature": 0,
    }

    try_num = 0
    while try_num < max_try_num:
        try_num += 1
        try:
            res_obj = chat(**data)
            # print("\nres_obj:" + str(res_obj) + "\n")
            if res_obj['choices'][0]['finish_reason'] == 'stop':
                answer = res_obj['choices'][0]['message']['content']
                return answer
        except:
            traceback.print_exc()
            time.sleep(3)
    return ''


if __name__ == '__main__':
    messages = [
        {
            "role": "system",
            "content": "你是NLP专家，对NLP的进展，尤其是大规模语言模型GPT、BERT等，非常了解"
        },
        {
            "role": "user",
            "content": '你工作多少年了？'
        },
        {
            "role": "assistant",
            "content": '我是一个AI语言模型，没有具体的工作经验和年限限制。我的设计和开发团队在不断地更新和优化我的算法和结构，'
                       '以更好地服务于用户。'
        },
        {
            "role": "user",
            "content": '你知道百度吗？'
        }
    ]
    print(json.dumps(messages, ensure_ascii=False))
    model_parameters = {
        # 'model': 'gpt-3.5-turbo-0613',
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'messages': messages,
    }
    result = chat(**model_parameters)
    print(result)

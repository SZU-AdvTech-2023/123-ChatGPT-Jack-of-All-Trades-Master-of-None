from typing import List
from typing import Dict
import traceback
import time

from wsd.data import wsd_data
from wsd.utils import chatgpt_client
from wsd.tools import word_analyzer


class WsdChatGPT:
    """

    """

    def __init__(self, start_tag: str = '<s>', end_tag: str = '</s>',
                 model: str = 'gpt-3.5-turbo-0301', max_tokens: int = 200, request_timeout: int = 10,
                 max_try_num: int = 3):
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.model = model
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.max_try_num = max_try_num

    def generate_context(self, word: str, pre_context: str, post_context: str):
        """

        :param word:
        :param pre_context:
        :param post_context:
        :return:
        """
        result = f'{pre_context} {self.start_tag} {word} {self.end_tag} {post_context}'
        return result

    @staticmethod
    def answer_engineer(keys: List[str], answer: str):
        """
        :param keys:
        :param answer:
        :return:
        """
        result = []
        for key in keys:
            if key in answer:
                result.append(key)
        return result

    @staticmethod
    def answer_engineer_2(senses, answer):
        """
        :param senses:
        :param answer:
        :return:
        """
        result = []
        for key, value in senses.items():
            if value in answer:
                result.append(key)
        return result

    def generate_prompt(self, word: str, context: str, senses: Dict, top_n: int = 1):
        """

        :param word:
        :param context:
        :param senses:
        :param top_n:
        :return:
        """
        if top_n == 1:
            result = """Which meaning of the word "{word}" between {start_tag} and {end_tag} in the following context is expressed: "{sentence}"
The meanings are as follows: {senses}.
Return only the key of the most relevant meaning.
                     """.format(word=word, sentence=context, senses=str(senses),
                                start_tag=self.start_tag, end_tag=self.end_tag)
#
#             result = """In the same way, I provide you with an English context : "{sentence}"
# The target word in the context, wrapped by tags {start_tag} and {end_tag}, is the word "{word}" that you need to pay attention to.
# I will provide you with all the possible senses of this target word as follows: {senses}
#
# Now please select the most appropriate sense option from the given list of word senses based on the provided context. Return only the key of the most relevant sense option.
#                      """.format(word=word, sentence=context, senses=str(senses),
#                                 start_tag=self.start_tag, end_tag=self.end_tag)

#             ordered_senses = "\n".join([f"{i + 1}. {sense}" for i, sense in enumerate(senses.values())])
#             result = """以下是给出的一个另外实例，你可以以相同的思考方式选择出目标词确切的词义：
#
# 英文语段：{sentence}
# 目标词：{word}
# 可能的词义选项：
# {ordered_senses}
#
# 请从这些词义选项中选出目标词{word}在给出的英文文段中的准确词义是哪一项，注意要给出完整的原英文词义选项本身。
# """.format(word=word, sentence=context, ordered_senses=ordered_senses)
#             result = """Which meaning of the word "{word}" between {start_tag} and {end_tag} in the following context is expressed: "{sentence}"
# The meanings are as follows: {senses}.
#
# The following methods may help you choose the correct meaning:
# 1. First, rate the semantic relatedness of each meaning to the given context.
# 2. If there are meanings that are semantically close and difficult to distinguish, you can try translating the target word into other languages to help align and judge.
# 3. Finally, select the meaning that best matches the target word and context.
#
# Make sure to return only the key of the most relevant meaning.
# """.format(word=word, sentence=context, senses=str(senses), start_tag=self.start_tag, end_tag=self.end_tag)
        else:
            result = """Which meaning of the word „{word}” between {start_tag} and {end_tag} in the following context is expressed: "{sentence}" The meanings are as follows: {senses}. 
                        Return only the keys of the {top_n} most relevant meaning.
                     """.format(word=word, sentence=context, senses=str(senses),
                                start_tag=self.start_tag, end_tag=self.end_tag,
                                top_n=top_n)
        return result

    def predict(self, word_info: Dict, glosses: Dict[str, str], top_n: int = 1,
                sense_key_prefix: str = 'key_'):
        # print("predicting...")
        """

        :param word_info:
        :param glosses:
        :param top_n:
        :param sense_key_prefix:
        :return:
        """
        word = word_info['word']
        pre_context = word_info['pre_context']
        post_context = word_info['post_context']
        sentence = self.generate_context(word, pre_context, post_context)

        if not glosses:
            return []

        if len(glosses) <= top_n:
            answer = [e['sense_id'] for e in glosses]
            return answer

        senses = {}
        for gloss in glosses:
            sense_id = gloss['sense_id']
            english_sense = gloss['sense']['first']
            senses[f'{sense_key_prefix}{sense_id}'] = english_sense

        prompt = self.generate_prompt(word, sentence, senses, top_n=top_n)
        # print("\nprompt: " + prompt)
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a english linguist."
#             },
#             {
#                 "role": "user",
#                 "content": """
# Now I have a specific contextual word sense disambiguation task for you. The task requirements are as follows:
#
# I provide you with an English context : "Beyond , the Edge becomes less well defined and is succeeded by steep slopes ,  <s> surmounted </s>  by inclining eastwards and following the county boundary and the watershed and keeping always to the height of land in surroundings of utter desolation yet of profound influence on the landscape , for three major rivers have their source hereabouts : the streams flowing east are tributaries of the River Swale , those to the west drain an area known as Eden Springs , the source of the River Eden , and a short distance to the south are the beginnings of the River Ure ."
# The target word in the context, wrapped by tags <s> and </s>, is the word "surmounted" that you need to pay attention to.
# I will provide you with all the possible senses of this target word as follows: {'surmount%2:33:00::': 'get on top of; deal with successfully', 'surmount%2:42:00::': 'be on top of', 'surmount%2:38:00::': 'reach the highest point of', 'surmount%2:33:01::': 'be or do something to a greater degree'}
#
# Please select the most appropriate sense option from the given list of word senses based on the provided context. Return only the key of the most relevant sense option.
#                 """
#             },
#             {
#                 "role": "assistant",
#                 "content": "The key of the most relevant sense option is 'surmount%2:38:00::'"
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]

#         messages = [
#             {"role": "user", "content": "请你配合我完成一个词义选择的任务，接下来我将详细描述任务要求"},
#             {"role": "assistant", "content": "当然可以，我很愿意帮助你完成词义选择的任务。请提供任务的详细描述，我将尽力配合你完成它。"},
#             {"role": "user", "content": "任务要求是：我提供一个英文语段，语段中有一个你需要找出准确词义的目标词（它由<s></s>标记），并且我将提供一些该目标词可能的几种词义解释的选项，请你给出其中最符合该文段语境的目标词词义选项"},
#             {"role": "assistant", "content": "明白了！请提供英文语段以及目标词的选项，我将尽力为你选择最符合语境的词义。请提供语段和选项，然后我会帮助你做出选择。"},
#             {"role": "user", "content": """以下是给出的一个实例：
# 英文语段：They can live in hot springs , or can be dredged from the deepest abyssal  <s> depths </s>  of the ocean ; the latter are grotesque gargoyles that seem to belong in the paintings of Hieronymus Bosch .
# 目标词：depths
# 可能的词义选项：
# 1. the extent downward or backward or inward
# 2. degree of psychological or intellectual profundity
# # 3. (usually plural) the deepest and most remote part
# # 4. (usually plural) a low moral state
# # 5. the intellectual ability to penetrate deeply into ideas
# # 6. the attribute or quality of being deep, strong, or intense
# #
# # 请从这些词义选项中选出目标词depths在给出的英文文段中的准确词义是哪一项，注意要给出完整的原英文词义选项本身。
# #             """},
# #             {"role": "user", "content": """你可以将语段和词义选项翻译成中文帮助理解和选择，分析步骤如下：
# #
# # 语段可以翻译成：”它们可以生活在温泉中，也可以从海洋最深的深渊中打捞出来；后者是怪诞的石像鬼，似乎属于Hieronymus Bosch的绘画作品。“
# # 词义选项可以翻译成：
# # 1. 向下、向后或向内的程度
# # 2. 心理或智力深度的程度
# # 3. （通常为复数）最深和最遥远的部分
# # 4. （通常为复数）道德状态低落
# # 5. 深入思想的智力能力
# # 6. 深厚、强烈或强烈的属性或品质
# #
# # 结合中文翻译和英文原文的语义信息，可以选出原英文文段中的目标词 depths 的准确英文词义选项是：(3) (usually plural) the deepest and most remote part（通常为复数形式，指最深且最遥远的部分）。
# #             """},
#             {"role": "assistant", "content": """在给出的英文语段中，目标词 depths 的完整的原英文词义选项是：(3) (usually plural) the deepest and most remote part（通常为复数形式，指最深且最遥远的部分）。
#             """},
#             {"role": "user", "content": prompt}
#         ]
        result = chatgpt_client.simple_chat(messages, model=self.model, max_tokens=self.max_tokens,
                                            request_timeout=self.request_timeout, max_try_num=self.max_try_num)
        print("\ncompletion: " + result)
        answer = self.answer_engineer(senses, result)
        # answer = self.answer_engineer_2(senses, result)
        answer = [e[len(sense_key_prefix):] for e in answer]
        # print(answer)
        return answer

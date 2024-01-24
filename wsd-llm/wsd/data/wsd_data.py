from typing import List
from typing import Dict
import os

from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn

from wsd.common import common_path
from wsd.utils import file_utils
from wsd.tools import nltk_wordnet

class Word:
    """

    """
    def __init__(self, word: str, lemma: str, pos: str, context: str, instance_id: str = '',
                 senses: Dict[str, str] = None):
        """

        :param word:
        :param lemma:
        :param pos:
        :param context: ...<t>word</t>...
        :param instance_id:
        :param senses:
        """
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.context = context
        self.instance_id = instance_id
        self.senses = senses

    def __repr__(self):
        return self.word


class Sentence:
    """

    """
    def __init__(self, words: List[Word], sentence_id):
        """

        :param words:
        """
        self.words = words
        self.sentence_id = sentence_id


class Document:
    """

    """
    def __init__(self, sentences: List[Sentence], doc_id: str):
        """

        :param docs:
        """
        self.sentences = sentences
        self.doc_id = doc_id


class Documents:
    """

    """
    def __init__(self, data_filepath: str, gold_filepath: str,
                 wsd_evaluation_framework: str, only_senses_with_proper_pos=True,
                 extra_context=False, max_context_len=8,
                 add_special_token=False):
        """

        :param data_filepath:
        :param gold_filepath:
        """
        # self.wn_senses_path = os.path.join(wsd_evaluation_framework, 'Data_Validation/candidatesWN30.txt')
        self.wn_senses_path = os.path.join(wsd_evaluation_framework, 'Data_Validation', 'candidatesWN30.txt')
        self.pos_converter = {'NOUN': 'n', 'PROPN': 'n', 'VERB': 'v', 'AUX': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.wn_senses = {}
        self.load_wn_senses()

        self.only_senses_with_proper_pos = only_senses_with_proper_pos
        self.add_special_token = add_special_token
        self.data_filepath = data_filepath
        self.gold_filepath = gold_filepath
        self.docs: List[Document] = []
        self.keys = {}
        self.load_data()
        self.load_key()

        self.extra_context = extra_context
        self.max_context_len = max_context_len

    def generate_key(self, lemma, pos):
        if pos in self.pos_converter.keys():
            pos = self.pos_converter[pos]
        key = '{}+{}'.format(lemma, pos)
        return key

    def load_wn_senses(self):
        with open(self.wn_senses_path, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip().split('\t')
                lemma = line[0]
                pos = line[1]
                senses = line[2:]
                key = self.generate_key(lemma, pos)
                self.wn_senses[key] = senses

    def get_all_instances(self):
        """

        :return:
        """
        result = []
        for doc in self.docs:
            for i, sentence in enumerate(doc.sentences):
                pre_sentence = '' if i == 0 else ' '.join([str(w) for w in doc.sentences[i-1].words])
                after_sentence = '' if i == len(doc.sentences) - 1 else ' '.join([str(w) for w in doc.sentences[i+1].words])
                for word in sentence.words:
                    word: Word = word
                    if self.extra_context and len(sentence.words) < self.max_context_len:
                        word.context = (pre_sentence + ' ' + word.context + ' ' + after_sentence).strip()
                    if word.instance_id:
                        result.append(word)
                    # if word.instance_id:
                    #     word.context = (pre_sentence + ' ' + word.context + ' ' + after_sentence).strip()
                    #     result.append(word)
        return result

    def get_pos_instances(self):
        """

        :return:
        """
        result = []
        for doc in self.docs:
            for i, sentence in enumerate(doc.sentences):
                tokens = [word.word for word in sentence.words]
                labels = [word.pos for word in sentence.words]
                result.append((tokens, labels))
        return result

    def load_data(self):
        text = file_utils.read_all_content(self.data_filepath)
        soup = BeautifulSoup(text, 'lxml')
        doc_tags = soup.find_all(name='text')
        num_of_instances_without_senses = 0
        instance_num = 0
        for doc_tag in doc_tags:
            doc_id = doc_tag['id']
            sentence_tags = doc_tag.find_all(name='sentence')
            sentences = []
            for sentence_tag in sentence_tags:
                sentence_id = sentence_tag['id']
                word_tags = sentence_tag.find_all()
                words = []
                for word_tag in word_tags:
                    original_word = word_tag.text
                    lemma = word_tag['lemma']
                    pos = word_tag['pos']
                    if word_tag.name == 'instance':
                        instance_id = word_tag['id']
                        if self.only_senses_with_proper_pos:
                            key = self.generate_key(lemma, pos)
                            if key in self.wn_senses:
                                sensekey_arr = self.wn_senses[key]
                                senses = {s: wn.lemma_from_key(s).synset().definition() for s in sensekey_arr}
                                instance_num += 1
                            else:
                                num_of_instances_without_senses += 1
                        else:
                            senses = nltk_wordnet.query_word_senses(original_word)
                    else:
                        instance_id = ''
                        senses = {}
                    word = Word(original_word, lemma, pos, '', instance_id, senses=senses)
                    words.append(word)
                words_str = [e.word for e in words]
                for i, word in enumerate(words):
                    left = words_str[: i]
                    right = words_str[i + 1:]
                    if self.add_special_token:
                        word.context = ' '.join(left + ['<target> ' + word.word + ' </target>'] + right)
                    else:
                        word.context = ' '.join(left + [word.word] + right)
                sentence = Sentence(words, sentence_id)
                sentences.append(sentence)
            doc = Document(sentences, doc_id)
            self.docs.append(doc)

        print('instance_num: %d' % instance_num)
        print('num_of_instances_without_senses: %d' % num_of_instances_without_senses)

    def load_key(self):
        lines = file_utils.read_all_lines(self.gold_filepath)
        for line in lines:
            parts = line.split()
            self.keys[parts[0]] = parts[1:]


if __name__ == '__main__':
    data_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                          'ALL.data.xml')
    key_filepath = os.path.join(common_path.project_dir, 'datasets/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/'
                                                          'ALL.gold.key.txt')
    docs = Documents(data_filepath, key_filepath)
    print('end')

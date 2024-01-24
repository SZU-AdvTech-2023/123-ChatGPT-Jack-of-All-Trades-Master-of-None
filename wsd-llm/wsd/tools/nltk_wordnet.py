"""
import nltk
nltk.download('wordnet')

online wordnet 3.1
http://wordnetweb.princeton.edu/perl/webwn?s=door&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=1&o3=&o4=&h=0000
"""

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print("Wordnet {}".format(wordnet.get_version()))


def gloss_from_sense_key(sense_key: str) -> str:
    return wordnet.lemma_from_key(sense_key).synset().definition()


def query_word_senses(lemma: str, pos: str = None):
    """
    NLTK库WordNet的使用方法实例 https://www.ngui.cc/el/1780093.html?action=onClick
    :param word:
    :param pos:
    :return:
    """
    result = {}
    if pos:
        synsets = wordnet.synsets(lemma, pos=pos)
    else:
        synsets = wordnet.synsets(lemma)
    for synset in synsets:
        keys = [e for e in synset.lemmas() if e.key()[:len(lemma) + 1] == (lemma + '%')]
        if not keys:
            # print('erroneous word keys: %s' % lemma)
            continue
        key = keys[0].key()
        definition = synset.definition()
        result[key] = definition
    return result


if __name__ == '__main__':
    synsets = query_word_senses('aas', 'n')
    print(synsets)

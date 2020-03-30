import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn, stopwords, sentiwordnet as swn
from nltk import word_tokenize, pos_tag, bigrams
import re
import spacy
nlp = spacy.load("en_core_web_sm")

# Utils
def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def xml_to_object(file):
    root = ET.parse(file).getroot()
    sentences = []
    for sentence in root:
        data = {"text": sentence.find('text').text, "aspects": []}
        if sentence.find('aspectTerms') != None:
            for term in sentence.find('aspectTerms'):
                data["aspects"].append(term.attrib)
        sentences.append(data)
    return sentences

def frequent_aspect_terms(dataset):
    sentences = xml_to_object("./datasets/" + dataset + "_train.xml")
    frequent = {}
    for sentence in sentences:
        aspects = sentence["aspects"]
        for aspect in aspects:
            terms = word_tokenize(re.sub(r'[^\w\s]','',aspect["term"]))
            for term in terms:
                if term not in set(stopwords.words('english')):
                    try:
                        frequent[term] += 1
                    except KeyError:
                        frequent[term] = 1
    return frequent

def parsing_tree_dict(sentence):
    doc = nlp(sentence)
    words = {}
    for token in doc:
        words[token.text] = {"dep": token.dep_,"head_text": token.head.text, "head_pos": token.head.pos_}
    return words

def dependency_tree_parsing(sentence):
    doc = nlp(sentence)
    vectors = {'nsubj': [], 'dobj': [], 'iobj': [], 'cop': [], 'conj': [], 'cc': []}
    for w in doc:
        try:
            vectors[w.dep_].append(w.text)
        except KeyError:
                continue
    return vectors

def spacy_lemma(word):
    doc = nlp(word)
    return "".join([token.text if token.lemma_ == '-PRON-' else token.lemma_ for token in doc])




# Sujeito (noun subject) nsubj = []
# Objeto direto (direct object) dobj = []
# Objeto Indireto (indirect object) iobj = []
# Copula cop = []
# Conjunção (conjunct) conj = []
# Conjunção coordenada (coordinate) cc = []
# # vectors = {'nsubj': [], 'dobj': [], 'iobj': [], 'cop': [], 'conj': [], 'cc': []}
# for w in doc:
#     try:
#         vectors[w.dep_].append(w)
#     except KeyError:
#             continue
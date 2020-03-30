from nltk import word_tokenize, pos_tag, bigrams
from nltk.stem.snowball import SnowballStemmer
from utils import *
import string

# OK
def lemma_word(word):
    if spacy_lemma(word) == '-PRON-':
        return (word,)
    return (spacy_lemma(word),)

# OK
def stem_word(word):
    return (SnowballStemmer("english", ignore_stopwords=True).stem(word),)

# OK
def wordnet_synsets(word):
    if pos_tag([word])[0][1] not in ['JJ','JJR', 'JJS','NN', 'NNS','NNP','NNPS', 'RB', 'RBR', 'RBS', 'VB','VBD','VBG','VBN', 'VBP','VBZ']:
        return (word,"NULL")
    diff_synsets = [s.lemmas()[0].name() for s in wn.synsets(word) if s.lemmas()[0].name() != word]
    eq_synsets = [s.lemmas()[0].name() for s in wn.synsets(word)]
    synset = diff_synsets + eq_synsets
    synset = synset[:2]
    synset = synset + ["NULL"]*(2 - len(synset))
    return tuple(synset)

# OK
def word_hypernyms(word):
    if pos_tag([word])[0][1] not in ['JJ','JJR', 'JJS','NN', 'NNS','NNP','NNPS', 'RB', 'RBR', 'RBS', 'VB','VBD','VBG','VBN', 'VBP','VBZ']:
        return (word,"NULL")
    if wn.synsets(word):
        hypernyms1 = wn.synsets(word)[0].hypernyms()
        hypernyms_names1 = [s.lemmas()[0].name() for s in hypernyms1]
        hypernyms_name1 = hypernyms_names1[0] if hypernyms_names1 else 'NULL'
        if hypernyms_name1 != 'NULL' and wn.synsets(hypernyms_name1):
            hypernyms2 = wn.synsets(hypernyms_name1)[0].hypernyms()
            hypernyms_names2 = [s.lemmas()[0].name() for s in hypernyms2]
            hypernyms_name2 = hypernyms_names2[0] if hypernyms_names2 else 'NULL'
        else:
            hypernyms_name2 = 'NULL'
        return (hypernyms_name1, hypernyms_name2)
    else:
        return ('NULL', 'NULL')
    
    

# OK
def antonymy(row):
    wn_pos = penn2morphy(row[1])
    synsets = wn.synsets(row[0], pos=wn_pos)

    if wn_pos == '':
        return ("NULL",)
    else:
        if synsets != [] and synsets[0].lemmas() != [] and synsets[0].lemmas()[0].antonyms() != []:
            return (synsets[0].lemmas()[0].antonyms()[0].name(),)
        else:
            return ("NULL",)

# OK
def stop_word(word):
    return ("1",) if word in set(stopwords.words('english')) else ("0",)

# OK
def is_frequent_aspect_term(word, frequent):
    try:
        return ("1",) if frequent[word] >= 4 else ("0",)
    except KeyError:
        return ("0",)

# OK
def dependency_trees(word, sentence):
    vectors = dependency_tree_parsing(sentence)
    nsubj = "1" if word in vectors["nsubj"] else "0"
    dobj = "1" if word in vectors["dobj"] else "0"
    iobj = "1" if word in vectors["iobj"] else "0"
    cop = "1" if word in vectors["cop"] else "0"
    conj = "1" if word in vectors["conj"] else "0"
    cc = "1" if word in vectors["cc"] else "0"
    return (nsubj, dobj, iobj, cop, conj, cc)

# 
# OK
def iob_labels(row, aspects):
    newRow = None
    if aspects == []:
        newRow = tuple("O")
    else:
        for aspect in aspects:
            terms = word_tokenize(re.sub(r'[^\w\s]','',aspect["term"]))
            i = 0
            for term in terms:
                if row[0] == term:
                    if i == 0:
                        newRow = tuple("B")
                    else:
                        newRow = tuple("I")
                if row[0] != term and newRow is None:
                    newRow = tuple("O")
                i = i + 1
    return newRow

# Not used
# F11
def synonymy(row):
    wn_pos = penn2morphy(row[1])
    if wn_pos == '':
        return (row[0],)
    else:
        if wn.synsets(row[0], pos=wn_pos) != []:
            return (wn.synsets(row[0], pos=wn_pos)[0].lemmas()[0].name(),)
        else:
            return ("NULL",)

# OK
def tokenize_feature(text):
    sent = word_tokenize(text)
    words = [] 
    for i, w in enumerate(sent):
        if i == len(sent)-1:
            words.append([w] + ["NULL"])
            wn.synsets("dog")
        else:
            words.append([w] + [sent[i+1]])
    return words

# OK
def pos_tag_sub(sent, sent2):
    for i, row in enumerate(sent):
        if i+1 < len(sent2):
            sent[i] = sent[i] + (sent[i+1][1],)
        else:
            sent[i] = sent[i] + ("NULL",)
    
    return sent

def negative_four(word):
    negative = ['dont', 'never', 'no', 'nothing', 'nowhere', 'noone', 'none', 'not', 'hasnt', 'hadnt', 'cant', 'couldnt', 'shouldnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'aint', 'scarcely', 'cannot']
    return ("1",) if word.lower() in negative else ("0",)

def positive_score(word):
    if len(list(swn.senti_synsets(word))) == 0:
        return ("0", )
    else:
        synset = list(swn.senti_synsets(word))[0]
        score = int(synset.pos_score()*5)
        return (str(score),)

def negative_score(word):
    if len(list(swn.senti_synsets(word))) == 0:
        return ("0", )
    else:
        synset = list(swn.senti_synsets(word))[0]
        score = int(synset.neg_score()*5)
        return (str(score),)

def invalid_char(word):
    return ("1",) if word in set(string.punctuation) else ("0",)

def distance_to_adj(idx, sent):
    dists = []
    for i, w in enumerate(sent):
        if w[1] in ["JJ", "JJR", "JJS"]:
            if idx < i:
                dists.append(i-idx)
            else:
                dists.append(idx-i)
    return (str(min(dists)),) if dists else ("NULL",)
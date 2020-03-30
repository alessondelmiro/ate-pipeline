from utils import *
from features import *
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
import time

# from preprocessing import *
# pipeline("laptops", "train")
# pipeline("laptops", "test")
# pipeline("restaurants", "train")
# pipeline("restaurants", "test")

# Bigrams only
# results('restaurants', 'macro') -> (0.8575778032232154, 0.7166340431707555, 0.7689280283714969, None)
# results('restaurants', 'micro') -> (0.9386238532110092, 0.9386238532110092, 0.9386238532110092, None)
# results('laptops', 'macro') -> (0.8873931106822234, 0.6612402317266337, 0.7373429378427998, None)
# results('laptops', 'micro') -> (0.9468932038834952, 0.9468932038834952, 0.9468932038834951, None)

# With B on the end
# results("restaurants_output",names,'macro') -> (0.8726096466465902, 0.7425738317823773, 0.7942374450872811, None)
# results("laptops_output",names,'macro') -> (0.9029513246375992, 0.6900966164206738, 0.7667987488150763, None)

# Trigrams + bigrams
# results('restaurants_output_trigram',names, 'macro') -> (0.8592786364895817, 0.6983991981312165, 0.7541613554320875, None)
# results('laptops_output_trigram',names, 'macro') -> (0.8788625224415275, 0.6364081205810861, 0.7142314066216047, None)

# Trigrams only
# results('laptops_output_trigram_only',names, 'macro') -> (0.868452040752505, 0.631255439784431, 0.7056897568780429, None)
# results('restaurants_output_trigram_only',names, 'macro') -> (0.84807052132276, 0.6849920016280456, 0.7387243611072428, None)


# names = ['token', 'pos', 'lemma', 'stem', 'isSuperlative', 'isComparative','lastFourNegative', 'positiveScore', 'negativeScore', 'nominalSubject', 'directObject', 'indirectObject', 'copula', 'conjunction', 'coordinatingConjunction', 'synonym1','synonym2', 'hypernymParent', 'hypernymGrandparent', 'antonym', 'isStopWord', 'isFrequentAte', 'iob', 'label']
def results(path, names, average = 'macro'):
    then = time.time()
    url =  path
    dataframe = pd.read_csv(url, names=names, sep='\t')
    y_true = np.array(dataframe.iob.to_list())
    y_pred = np.array(dataframe.label.to_list())
    now = time.time()
    print("Execution time", now - then, "s")
    return precision_recall_fscore_support(y_true, y_pred, average = average)


def pipeline(dataset, dataset_type):
    then = time.time()
    file = "./datasets/" + dataset + "_" + dataset_type + ".xml"
    file_out = "./crf_files/" + dataset + "_" + dataset_type
    out = open(file_out, "w", encoding="utf-8")
    sentences = xml_to_object(file)
    frequent = frequent_aspect_terms(dataset)
    for sentence in sentences:
        text = sentence["text"]
        aspects = sentence["aspects"]
        text = re.sub(r'[^\w\s]','',text.strip())
        words = word_tokenize(text) # F1
        sent = pos_tag(words) #F2
        for i, row in enumerate(sent):
            lemma = lemma_word(row[0]) #F3
            stem = stem_word(row[0]) #F4
            sup = ("1",) if row[1] == "JJS" or row[1] == "RBS" else ("0",) #F5
            comp = ("1",) if row[1] == "JJR" or row[1] == "RBR" else ("0",) #F6
            neg = negative_four(sent[i-4][0]) if i > 3 else ("0",) #F7
            pscore = positive_score(row[0]) #F8
            nscore = negative_score(row[0]) #F9
            dep = dependency_trees(row[0], text) #F10 a 15
            syn = wordnet_synsets(row[0]) #F16 e 17
            hyp = word_hypernyms(row[0]) #F18 e 19
            ant = antonymy(row) #F20
            stp = stop_word(row[0]) #F21
            freq = is_frequent_aspect_term(row[0], frequent) #F22
            iob = iob_labels(row, aspects) #F23
            sent[i] = sent[i] + lemma + stem + sup + comp + neg + pscore + nscore + dep + syn + hyp + ant + stp + freq + iob
            out.write("\t".join(sent[i]) + "\n")
        out.write("\n")
    out.close()
    now = time.time()
    print("Execution time", now - then, "s")

# sent = [('But', 'CC', 'DT'), ('the', 'DT', 'NN'), ('staff', 'NN', 'VBD'), ('was', 'VBD', 'RB'), ('so', 'RB', 'JJ'), ('horrible', 'JJ', 'TO'), ('to', 'TO', 'PRP'), ('us', 'PRP', 'NULL')]

# import pandas as pd
# import numpy as np
# url = "./crf_files/restaurants_output_test.csv"

# dataframe = pd.read_csv(url, names=names, sep='\t')

# from sklearn.metrics import classification_report
# iob=dataframe.BIO.to_list()
# klass=dataframe.Class.to_list()
# y_true = np.array(iob)
# y_pred = np.array(klass)
# precision_recall_fscore_support(y_true, y_pred, average='macro')
# 
# precision_recall_fscore_support(y_true, y_pred, average='micro')
# 

def ablation(data):
    results_file = open(f'results_{data}.csv', 'w+')
    names = ['token', 'pos', 'lemma', 'stem', 'isSuperlative', 'isComparative','lastFourNegative', 'positiveScore', 'negativeScore', 'nominalSubject', 'directObject', 'indirectObject', 'copula', 'conjunction', 'coordinatingConjunction', 'synonym1','synonym2', 'hypernymParent', 'hypernymGrandparent', 'antonym', 'isStopWord', 'isFrequentAte']
    for name in names:
        if name not in ['token']:
            ind = str(names.index(name))
            template_file = open('crf_files/templates/ablation/template_'+name, 'w+')
            template_file.write('U0:%x[0,0]\n')
            template_file.write(f"U{ind}:%x[0,{ind}]\n")
            template_file.close()
            os.system(f'crf_learn crf_files/templates/ablation/template_{name} crf_files/data/{data}_train crf_files/models/ablation/{data}_{name}_model')
            os.system(f'crf_test -m crf_files/models/ablation/{data}_{name}_model crf_files/data/{data}_test > crf_files/output/ablation/{data}_output_{name}')
            result = results(f'ablation/{data}_output_{name}')
            result = [str(float(n)) for n in list(result)[:-1]]
            results_file.write(name + "\t" + "\t".join(result) + "\n")
    
    results_file.close()
import pandas as pd
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
import pathlib
import os
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def pos(sentence):

    '''
    Pos labels one single sentences by using Part-Of-Speech Tagging
    :param sentence: (STR)
    :return:
    '''

    sentence = sent_tokenize(sentence)[0]
    doc = nlp(sentence)
    singular = 0
    plural = 0
    neutral = 0
    for idx, token in enumerate(doc):
        if token.tag_ == "NN" or str(token).lower() in ["i","he","she", "it", "myself"]:
            singular += 1
        if token.tag_ == "NNS" or str(token).lower() in ["we", "they", "themselves", "themself", "ourselves", "ourself"]:
            plural += 1

    if singular > plural:
        #print("SINGULAR: ",sentence)
        return [1, 0, 0]
    if singular < plural:
        #print("PLURAL: ",sentence)
        return [0, 0, 1]
    if singular == plural:
        #print("NEUTRAL: ",sentence)
        return [0, 1, 0]


def labeling(sentences, file_name):
    labels = []
    for s in tqdm(sentences[:100]):
        label = pos(s)
        labels.append(label)

    labels = np.asarray(labels)
    df_labels = pd.DataFrame({'Sentence': sentences[:100],
                              'Singular': labels[:, 0],
                              'Neutral': labels[:, 1],
                              'Plural': labels[:, 2]})

    df_labels.to_csv(pathlib.Path('results', f'{file_name}-pron-pred'), index=False)

import en_core_web_sm
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nlp = en_core_web_sm.load()

'''
This script generates the verb tense labels for the dataset and it creates train, dev, test with sentiment and 
verb tense
'''


def find_indexes(labels):
    labels = np.asarray(labels)
    return np.where(labels == -1)[0].tolist()


# drop sentences from the df that are neither past nor present
def drop_indexes(df, labels):
    indexes = find_indexes(labels)
    df = df.drop(df.index[indexes])
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    return df


def hot_one(labels):
    l = []

    for i in labels:
        if i == 0:
            l.append([1, 0])
        if i == 1:
            l.append([0, 1])
        if i == -1:
            l.append([0, 0])
    l = np.asarray(l)
    labels = pd.DataFrame({'Present': l[:, 0], 'Past': l[:, 1]})
    return labels


def pos(sentence):
    sentence = sent_tokenize(sentence)[0]
    doc = nlp(sentence)
    present = 0
    past = 0
    for idx, token in enumerate(doc):
        if token.tag_ == "VBP" or token.tag_ == "VBZ":
            present += 1
        if token.tag_ == "VBD":
            past += 1
    if present > past:
        return "pres", sentence
    if past > present:
        return "pas", sentence
    else:
        return "no", sentence


def labeling(sentences):
    labels = []
    for s in tqdm(sentences):
        tag, sentence = pos(s)
        if tag == "pres":
            labels.append(0)
        if tag == "pas":
            labels.append(1)
        if tag == 'no':
            labels.append(-1)

    return labels


def generate_tense_label_files(sentences):
    tense_labels = labeling(sentences[:100])
    labels = hot_one(tense_labels[:100])
    return labels

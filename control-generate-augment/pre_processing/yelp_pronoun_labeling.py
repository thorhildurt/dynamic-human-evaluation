import pandas as pd
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
import pathlib
import os
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
data_dir = pathlib.Path('..', 'data', 'yelp')
pronoun_dir = pathlib.Path(data_dir, 'pronoun_data')

'''
This script generates the pronouns for the data set (plural, singular or neutral)
'''


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


def labeling(file, split):
    data = pd.read_csv(pathlib.Path(data_dir, file))
    labels = []
    for i in tqdm(range(data.shape[0])):
        s = data.review.iloc[i]
        label = pos(s)
        labels.append(label)

    labels = np.asarray(labels)
    df_labels = pd.DataFrame({'Singular': labels[:, 0],
                              'Neutral': labels[:, 1],
                              'Plural': labels[:, 2]})

    df_labels.to_csv(pathlib.Path(pronoun_dir, "y_yelp_{0}.csv".format(split)), index=False)
    data.to_csv(pathlib.Path(pronoun_dir, f"yelp_{split}.csv", index=False))


def generate_pron_label_files(num_sentences=-1):
    yelp_train = "yelp_train.csv"
    yelp_valid = "yelp_valid.csv"
    yelp_test = "yelp_test.csv"

    if not os.path.exists(pronoun_dir):
        os.makedirs(pronoun_dir)

    print("Pronoun labeling train dataset")
    labeling(yelp_train, "train")
    print("Pronoun labeling valid dataset")
    labeling(yelp_valid, "valid")
    print("Pronoun labeling test dataset")
    labeling(yelp_test, "test")

    print("Processing complete\n")


def main():
    generate_pron_label_files()


if __name__ == "__main__":
    main()


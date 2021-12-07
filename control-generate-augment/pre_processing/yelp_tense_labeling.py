import os
import pathlib
import en_core_web_sm
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.utils import shuffle
from tqdm import tqdm
import yelp_file_consts as yf

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


def labeling(dataset):
    labels = []
    unlabeled = []
    for i in tqdm(range(dataset.shape[0])):
        sentence = dataset.review.iloc[i]
        tag, sentence = pos(sentence)
        if tag == "pres":
            labels.append(0)
        if tag == "pas":
            labels.append(1)
        if tag == 'no':
            labels.append(-1)

    return labels


# generate present and past label for sentences
# drop sentences that have neither present nor past label
def generate_tense_label_files():

    if not os.path.exists(yf.tense_data_dir):
        os.makedirs(yf.tense_data_dir)

    print("Tense labeling train dataset")
    # For training
    data_train = pd.read_csv(pathlib.Path(yf.data_dir, yf.yelp_train))

    labels_tense_train = labeling(data_train)
    X_train = drop_indexes(data_train, labels_tense_train)
    y_train = hot_one(labels_tense_train)
    assert X_train.shape[0] == y_train.shape[0]

    X_train.to_csv(pathlib.Path(yf.tense_data_dir, yf.proxy_yelp_train), index=False)
    y_train.to_csv(pathlib.Path(yf.tense_data_dir, yf.proxy_y_yelp_train), index=False)
    pd.DataFrame(labels_tense_train).to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_train_complete), index=False)

    print("Tense labeling valid dataset")
    # For validation
    data_dev = pd.read_csv(pathlib.Path(yf.data_dir, yf.yelp_valid))

    labels_tense_dev = labeling(data_dev)
    X_dev = drop_indexes(data_dev, labels_tense_dev)
    y_dev = hot_one(labels_tense_dev)
    assert X_dev.shape[0] == y_dev.shape[0]
    X_dev, y_dev = shuffle(X_dev, y_dev)

    X_dev.to_csv(pathlib.Path(yf.tense_data_dir, yf.yelp_valid), index=False)
    y_dev.to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_valid), index=False)
    pd.DataFrame(labels_tense_dev).to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_valid_complete), index=False)

    print("Tense labeling test dataset")
    # For testing
    data_test = pd.read_csv(pathlib.Path(yf.data_dir, yf.yelp_test))

    labels_tense_test = labeling(data_test)
    X_test = drop_indexes(data_test, labels_tense_test)
    y_test = hot_one(labels_tense_test)
    assert X_test.shape[0] == y_test.shape[0]

    X_test.to_csv(pathlib.Path(yf.tense_data_dir, yf.yelp_test), index=False)
    y_test.to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_test), index=False)
    pd.DataFrame(labels_tense_test).to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_test_complete), index=False)
    print("Processing complete\n")


def main():
    generate_tense_label_files()


if __name__ == "__main__":
    main()

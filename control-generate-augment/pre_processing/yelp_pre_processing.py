import os
import pathlib
import en_core_web_sm
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.utils import shuffle
from tqdm import tqdm
import yelp_sentiment_labeling as ysl
import yelp_pronoun_labeling as ypl
import yelp_tense_labeling as ytl
import yelp_file_consts as yf

# set to -1 for the entire corpus
num_sentences = 100


def concat_labels_for_multi_attribute_control():
    # start by reading all data from the tense attribute directory
    X_train = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.proxy_yelp_train))
    y_train = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.proxy_y_yelp_train))
    X_test = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.yelp_test))
    y_test = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_test))
    X_dev = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.yelp_valid))
    y_dev = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_valid))

    # concatenate yelp training data and yelp test data
    # and corresponding tense labels for training and test generated above
    new_X_train = pd.concat([X_train, X_test])
    new_y_tense_train = pd.concat([y_train, y_test])

    new_X_train.to_csv(pathlib.Path(yf.tense_data_dir, yf.yelp_train), index=False)
    new_y_tense_train.to_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_train), index=False)

    # read the sentiment labels for train and test data
    sentiment_tense_data_dir = pathlib.Path(yf.data_dir, 'sentiment_tense')
    if not os.path.exists(sentiment_tense_data_dir):
        os.makedirs(sentiment_tense_data_dir)

    print("Concat tense labels and sentiment labels")
    labels_sent_train = pd.read_csv(pathlib.Path(yf.data_dir, yf.y_yelp_train))
    labels_tense_train = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_train_complete))
    y_sent_train = ytl.drop_indexes(labels_sent_train, labels_tense_train)

    labels_sent_test = pd.read_csv(pathlib.Path(yf.data_dir, yf.y_yelp_test))
    labels_tense_test = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_test_complete))
    y_sent_test = ytl.drop_indexes(labels_sent_test, labels_tense_test)

    new_y_sent_train = pd.concat([y_sent_train, y_sent_test])

    print("new_y_tense_train", new_y_tense_train.shape)
    print("new_y_sent_train", new_y_sent_train.shape)

    new_y_sent_tense_train = pd.concat([new_y_sent_train, new_y_tense_train], axis=1)
    new_X_train.to_csv(pathlib.Path(sentiment_tense_data_dir, yf.yelp_train), index=False)
    new_y_sent_tense_train.to_csv(pathlib.Path(sentiment_tense_data_dir, yf.y_yelp_train), index=False)

    # read sentiment labels for validation data and concatenate with tense labels
    labels_sent_dev = pd.read_csv(pathlib.Path(yf.data_dir, yf.y_yelp_valid))
    labels_tense_valid = pd.read_csv(pathlib.Path(yf.tense_data_dir, yf.y_yelp_valid_complete))
    y_sent_dev = ytl.drop_indexes(labels_sent_dev, labels_tense_valid)

    # copy validation data to new directory
    X_dev.to_csv(pathlib.Path(sentiment_tense_data_dir, yf.yelp_valid), index=False)

    new_y_sent_tense_dev = pd.concat([y_sent_dev, y_dev], axis=1)
    new_y_sent_tense_dev.to_csv(pathlib.Path(sentiment_tense_data_dir, yf.y_yelp_valid), index=False)

    # concat tense, sentiment and pronoun labels for training data (where we also merge test data)
    pron_sentiment_tense_data_dir = pathlib.Path(yf.data_dir, 'pron_sentiment_tense')
    if not os.path.exists(pron_sentiment_tense_data_dir):
        os.makedirs(pron_sentiment_tense_data_dir)

    print("Concat tense labels, sentiment labels, and pronoun labels")
    labels_pron_train = pd.read_csv(pathlib.Path(yf.pron_data_dir, yf.y_yelp_train))
    y_pron_train = ytl.drop_indexes(labels_pron_train, labels_tense_train)

    labels_pron_test = pd.read_csv(pathlib.Path(yf.pron_data_dir, yf.y_yelp_test))
    y_pron_test = ytl.drop_indexes(labels_pron_test, labels_tense_test)

    new_y_pron_train = pd.concat([y_pron_train, y_pron_test])

    print("new_y_sent_train", new_y_sent_train.shape)
    print("new_y_tense_train", new_y_tense_train.shape)
    print("new_y_pron_train", new_y_pron_train.shape)

    new_y_tense_sent_pron_train = pd.concat([new_y_pron_train, new_y_sent_tense_train], axis=1)
    new_X_train.to_csv(pathlib.Path(pron_sentiment_tense_data_dir, yf.yelp_train), index=False)
    new_y_tense_sent_pron_train.to_csv(pathlib.Path(pron_sentiment_tense_data_dir, yf.y_yelp_train), index=False)

    # concat tense, sentiment and pronoun labels for validation data
    labels_pron_valid = pd.read_csv(pathlib.Path(yf.pron_data_dir, yf.y_yelp_valid))
    y_pron_dev = ytl.drop_indexes(labels_pron_valid, labels_tense_valid)
    X_dev.to_csv(pathlib.Path(pron_sentiment_tense_data_dir, yf.yelp_valid), index=False)

    new_y_tense_sent_pron_dev = pd.concat([y_pron_dev, new_y_sent_tense_dev], axis=1)
    new_y_tense_sent_pron_dev.to_csv(pathlib.Path(pron_sentiment_tense_data_dir, yf.y_yelp_valid), index=False)


# The sentiment labels are extracted from the
# source data files downloaded here:
# https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp
# The pronoun labeling and tense labeling depend
# on the csv data files generated when extracting
# the sentiment labels from the source data. Thus
# if we only want to train the mode with small amount of data we
# can specify the required amount of sentences in the below function
def main():
    # run yelp_sentiment_labeling
    ysl.generate_sent_label_files(num_sentences)

    # run yelp_pronoun_labeling
    ypl.generate_pron_label_files()

    # run tense labeling
    ytl.generate_tense_label_files()

    # concatenate labels together
    concat_labels_for_multi_attribute_control()


if __name__ == "__main__":
    main()

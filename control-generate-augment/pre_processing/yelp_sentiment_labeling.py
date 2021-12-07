'''
  Title: Linguistic Style-Transfersource code
  Author: Vineet John, Gaurav
  Date: 2021
  Comment: This code is based on the implementation script linguistic_style_transfer_model/corpus_adapters/yelp_corpus_adapter.py
  Code repository: https://github.com/vineetjohn/linguistic-style-transfer
'''

import re
import pathlib
import csv
import os
import yelp_file_consts as yf
import pandas as pd

data_dir = pathlib.Path('..', 'data', 'yelp')
sent_dir = pathlib.Path(data_dir, 'sentiment_data')

train_pos_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.train.1')
train_neg_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.train.0')
dev_pos_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.dev.1')
dev_neg_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.dev.0')
test_pos_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.test.1')
test_neg_reviews_file_path = pathlib.Path(yf.data_dir, 'sentiment.test.0')

train_csv_file_path = pathlib.Path(yf.data_dir, yf.yelp_train)
train_labels_file_path = pathlib.Path(yf.data_dir, yf.y_yelp_train)
valid_csv_file_path = pathlib.Path(yf.data_dir, yf.yelp_valid)
valid_labels_file_path = pathlib.Path(yf.data_dir, yf.y_yelp_valid)
test_csv_file_path = pathlib.Path(yf.data_dir, yf.yelp_test)
test_labels_file_path = pathlib.Path(yf.data_dir, yf.y_yelp_test)


def clean_text(string):
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r'\d+', "number", string)
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


def generate_sent_label_files(num_sentences=-1):
    if not os.path.exists(sent_dir):
        os.makedirs(sent_dir)

    print("Processing of base data with corresponding sentiment labels\n")
    print("Writing train dataset")
    with open(train_csv_file_path, 'w', newline='') as csv_file, open(train_labels_file_path, 'w', newline='') as labels_file:
        train_writer = csv.writer(csv_file)
        label_writer = csv.writer(labels_file)
        train_writer.writerow(["review"])
        label_writer.writerow(["Negative", "Positive"])

        # for testing and debugging
        if num_sentences > 0:
            cnt_pos = 0
            cnt_neg = 0
            max_sentences = num_sentences / 2
            with open(train_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_pos > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    train_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
                    cnt_pos += 1

            with open(train_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_neg > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    train_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])
                    cnt_neg += 1
        else:
            with open(train_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    train_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
            with open(train_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    train_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])

    print("Writing validation dataset")
    with open(valid_csv_file_path, 'w') as csv_file, open(valid_labels_file_path, 'w') as labels_file:
        val_writer = csv.writer(csv_file)
        label_writer = csv.writer(labels_file)
        val_writer.writerow(["review"])
        label_writer.writerow(["Negative", "Positive"])

        # for testing and debugging
        if num_sentences > 0:
            cnt_pos = 0
            cnt_neg = 0
            max_sentences = num_sentences / 2
            with open(dev_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_pos > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    val_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
                    cnt_pos += 1
            with open(dev_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_neg > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    val_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])
                    cnt_neg += 1
        else:
            with open(dev_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    val_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
            with open(dev_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    val_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])

    print("Writing test dataset")
    with open(test_csv_file_path, 'w') as csv_file, open(test_labels_file_path, 'w') as labels_file:
        test_writer = csv.writer(csv_file)
        label_writer = csv.writer(labels_file)
        test_writer.writerow(["review"])
        label_writer.writerow(["Negative", "Positive"])
        # for testing and debugging
        if num_sentences > 0:
            cnt_pos = 0
            cnt_neg = 0
            max_sentences = num_sentences / 2
            with open(test_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_pos > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    test_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
                    cnt_pos += 1
            with open(test_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if cnt_neg > max_sentences:
                        continue
                    if clean_text(line) == "":
                        continue
                    test_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])
                    cnt_neg += 1
        else:
            with open(test_pos_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    test_writer.writerow([clean_text(line)])
                    label_writer.writerow(["0", "1"])
            with open(test_neg_reviews_file_path, 'r') as reviews_file:
                for line in reviews_file:
                    if clean_text(line) == "":
                        continue
                    test_writer.writerow([clean_text(line)])
                    label_writer.writerow(["1", "0"])

    print("Processing complete\n")

    print("Copy files to sentiment data directory")
    splits = ['train', 'valid', 'test']
    for split in splits:
        if split == splits[0]:
            data = pd.read_csv(train_csv_file_path)
            labels = pd.read_csv(train_labels_file_path)
        elif split == splits[1]:
            data = pd.read_csv(valid_csv_file_path)
            labels = pd.read_csv(valid_labels_file_path)
        elif split == splits[2]:
            data = pd.read_csv(test_csv_file_path)
            labels = pd.read_csv(test_labels_file_path)
        data.to_csv(pathlib.Path(sent_dir, f"yelp_{split}.csv"), index=False)
        labels.to_csv(pathlib.Path(sent_dir, f"y_yelp_{split}.csv"), index=False)


def main():
    generate_sent_label_files()


if __name__ == "__main__":
    main()
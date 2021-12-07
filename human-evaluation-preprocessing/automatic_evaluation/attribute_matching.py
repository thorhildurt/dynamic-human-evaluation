import argparse
import pathlib
import pandas as pd
import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import random
import numpy as np
from textCNN import CNN
import torch.optim as optim
import time
import spacy
import os
import csv
import pronoun_classification as pc
import sentiment_classification as sc
import tense_classification as tc
import os.path
nlp = spacy.load('en_core_web_sm')

CLASSIFY_SENT = True
CLASSIFY_PRON = True
CLASSIFY_TENSE = True


def classify_generated_sentences(cuda, sentences, result_path, file_name):

    # sentiment classification
    if CLASSIFY_SENT:
        print("Predicting sentiment...")
        sentiment_predictions = sc.load_model_and_predict(cuda, sentences)
        result_file = pathlib.Path(result_path, f'{file_name}-sent-pred')
        with open(result_file, 'w') as txt_file:
            writer = csv.writer(txt_file)
            writer.writerow(["sentence", "sentiment"])
            for s in sentiment_predictions:
                writer.writerow([s[1], s[2]])
    # pronoun classification
    if CLASSIFY_PRON:
        print("Predicting personal pronoun...")
        pc.labeling(sentences, result_path, file_name)
    # tense classification
    if CLASSIFY_TENSE:
        print("Predicting verb tense...")
        tense_labels = tc.generate_tense_label_files(sentences)
        labels = np.asarray(tense_labels)
        df_labels = pd.DataFrame({'Sentence': sentences,
                                  'Present': labels[:, 0],
                                  'Past': labels[:, 1]})

        result_file = pathlib.Path(result_path, f'{file_name}-tense-pred')
        df_labels.to_csv(result_file, index=False)


def attribute_matching(generated_data, file_name):
    print(f"Attribute matching: {file_name}")
    num_sentences = len(generated_data)
    print(f"Number of generated sentences: {num_sentences}")

    # sentiment attribute matching
    sent_pred_results = pd.read_csv(pathlib.Path('generated_data_predictions', f'{file_name}-sent-pred'))
    label_match_cnt = 0
    pos_cnt = 0
    neg_cnt = 0
    for i in range(num_sentences):
        textcnn_label = 0
        attribute_label = 0
        assert(sent_pred_results.sentence.iloc[i] == generated_data.sentence.iloc[i])
        label_softmax = sent_pred_results.sentiment.iloc[i]
        if label_softmax > 0.5:
            textcnn_label = 1
        label_gen_data = generated_data.label.iloc[i]
        if 'Pos' in label_gen_data:
            attribute_label = 1

        if textcnn_label == 1 and attribute_label == 1:
            label_match_cnt += 1
            pos_cnt += 1
        elif textcnn_label == 0 and attribute_label == 0:
            label_match_cnt += 1
            neg_cnt += 1
        """
        else:
            print(f'attribute label: {attribute_label} - textcnn label: {textcnn_label}')
            print(f'{generated_data.sentence.iloc[i]} - {sent_pred_results.sentence.iloc[i]}')
        """
    print(f'Positive-match-count={pos_cnt}, negative-match-count={neg_cnt}')
    print(f'{pos_cnt+neg_cnt}/{num_sentences}')
    sent_percentage = label_match_cnt / num_sentences
    print("Sentiment", sent_percentage)

    # Pronoun attribute matching
    pron_pred_results = pd.read_csv(pathlib.Path('generated_data_predictions', f'{file_name}-pron-pred'))
    sing_cnt = 0
    pron_cnt = 0
    plural_cnt = 0
    for i in range(num_sentences):
        assert (pron_pred_results.Sentence.iloc[i] == generated_data.sentence.iloc[i])
        gen_data_label = generated_data.label.iloc[i]
        is_singular = pron_pred_results.Singular.iloc[i]
        if is_singular and 'Singular' in gen_data_label:
            sing_cnt += 1
            continue
        is_neutral = pron_pred_results.Neutral.iloc[i]
        if is_neutral and 'Neutral' in gen_data_label:
            pron_cnt += 1
            continue
        is_plural = pron_pred_results.Plural.iloc[i]
        if is_plural and 'Plural' in gen_data_label:
            plural_cnt += 1
            continue
        '''
        print(f'{generated_data.sentence.iloc[i]} - label: {gen_data_label}, '
              f'predictions: Singular - {pron_pred_results.Singular.iloc[i]}, Neutral - {pron_pred_results.Neutral.iloc[i]}, '
              f'Plural - {pron_pred_results.Plural.iloc[i]}')
        '''
    print(f'Singular-match-count={sing_cnt}, neutral-match-count={pron_cnt}, plural-match-count={plural_cnt}')
    print(f'{sing_cnt + pron_cnt + plural_cnt}/{num_sentences}')
    pronoun_percentage = (sing_cnt+pron_cnt+plural_cnt) / num_sentences
    print("Pronoun", pronoun_percentage)

    # Tense attribute matching
    tense_pred_results = pd.read_csv(pathlib.Path('generated_data_predictions', f'{file_name}-tense-pred'))
    pres_cnt = 0
    past_cnt = 0
    for i in range(num_sentences):
        assert (tense_pred_results.Sentence.iloc[i] == generated_data.sentence.iloc[i])
        gen_data_label = generated_data.label.iloc[i]
        is_present = tense_pred_results.Present.iloc[i]
        if is_present and 'Pres' in gen_data_label:
            pres_cnt += 1
            continue
        is_past = tense_pred_results.Past.iloc[i]
        if is_past and 'Past' in gen_data_label:
            past_cnt += 1
            continue
        """
        print(f'{generated_data.sentence.iloc[i]} - label: {gen_data_label}, '
              f'predictions: Present - {tense_pred_results.Present.iloc[i]}, Past - {tense_pred_results.Past.iloc[i]},')
        """
    print(f'Present-match-count={pres_cnt}, Past-match-count={past_cnt}')
    print(f'{pres_cnt + past_cnt}/{num_sentences}')
    tense_perentage = (pres_cnt + past_cnt) / num_sentences
    print("Tense", tense_perentage)

    # Store the results
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    columns = ['date', 'file_name', 'sent_match', 'tense_match', 'pron_match']
    values = [ts, file_name, sent_percentage, tense_perentage, pronoun_percentage]
    results = pd.DataFrame([values], columns=columns)

    if os.path.isfile('attribute_matching_results.csv'):
        df = pd.read_csv("attribute_matching_results.csv")
        df = pd.concat([df, results])
        df = df.set_index('date')
        df.to_csv("attribute_matching_results.csv")
    else:
        df = results.set_index('date')
        df.to_csv("attribute_matching_results.csv")


if __name__ == "__main__":
    # default file_name while testing
    #file_name = 'SentTensNumber'

    #file_name = '2021-Apr-27-09:53:10_6000_1'
    #file_name = '2021-Apr-30-09:22:17_6000_1'
    #file_name = '2021-Apr-26-12:28:06_6000_1'
    file_name = '2021-Apr-27-09:51:01_6000_1'
    file_name = '2021-Apr-30-11:44:46_6000_1'

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=file_name)
    parser.add_argument('--gpu', type=str, default='3')
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    # set desired gpu id for the spaceml cluster
    cuda = torch.device('cuda:' + args.gpu)

    # verify results folder for the predictions made on the generated data
    file_path = pathlib.Path('..', 'generated_output', f'{args.file}.txt')
    result_path = pathlib.Path('generated_data_predictions')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # read in the file we want to evaluate via attribute matching
    generated_data = pd.read_csv(file_path)

    classify_generated_sentences(cuda, generated_data['sentence'], result_path, file_name)
    attribute_matching(generated_data, file_name)

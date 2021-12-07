import argparse
import pathlib
import os
import pandas as pd
import re
from random import randrange
import random
import time

ATTR_CONTROL = 'multi_attr'
E2E = 'e2e'


class Sentence:
    def __init__(self, sentence="", label="", model_name="", clean_text=True):
        self.gen_sentence = sentence
        self.sentence = sentence
        self.label = label
        self.model_name = model_name
        if clean_text:
            self.clean_sentence()

    def clean_sentence(self):
        placeholder = "_num_"
        self.remove_tags()
        if placeholder in self.sentence:
            self.generate_random_num()
        self.fix_period_points()
        self.capitalise_sentence()

    def remove_tags(self):
        sentence_without_tags = re.sub('<.*?>', '', self.sentence)
        sentence_without_tags = sentence_without_tags.rstrip()
        if sentence_without_tags == "":
            return sentence_without_tags
        sentence_without_tags = ''.join((sentence_without_tags, '.'))
        self.sentence = sentence_without_tags

    def fix_period_points(self):
        sentence = re.sub('\.!', '.', self.sentence)
        sentence = re.sub('!\.', '.', sentence)
        sentence = re.sub(' \.', '.', sentence)

        self.sentence = sentence

    def generate_random_num(self):
        placeholder = "_num_"
        random_num = randrange(10, 15)
        sentence = re.sub(placeholder, str(random_num), self.sentence)
        self.sentence = sentence

    def capitalise_sentence(self):
        sentence = self.sentence
        self.sentence = sentence.capitalize()

    def is_empty(self):
        if self.sentence == "":
            return True
        else:
            return False

    def is_one_word(self):
        words = self.sentence.split()
        if len(words) <= 1:
            return True
        else:
            return False


def create_sentence_objects(dataframe, model_name, label):
    sentences = []

    for i in range(len(dataframe)):
        key = dataframe.sentence.iloc[i]
        sentence = Sentence(sentence=key,
                            label=dataframe.label.iloc[i],
                            model_name=model_name)

        if not sentence.is_empty() and not sentence.is_one_word():
            sentences.append(sentence)

    stats = {'model': model_name, 'label': label, 'num_sentences': len(sentences)}
    return sentences, stats


def remove_redundant_sentences(df_sentences, model_name):
    print('=== Sentence redundancy analysis ===')
    print(f'Sentences from {model_name}:')
    sentence_dict = {}
    updated_df = []
    for idx in range(len(df_sentences)):
        key = df_sentences.sentence.iloc[idx]
        if key in sentence_dict:
            sentence_dict[key] += 1
        else:
            sentence_dict[key] = 1
            updated_df.append(df_sentences.iloc[idx])
    df = pd.DataFrame(updated_df)
    num_redundant_sent = 0
    sentences_to_remove = 0
    for key in sentence_dict.keys():
        if sentence_dict[key] > 1:
            # print(f'[{key}] - freq: {sentence_dict[key]}')
            num_redundant_sent += 1
            sentences_to_remove += (sentence_dict[key] - 1)

    print(f'- Total {num_redundant_sent}/{len(sentence_dict)} for all distinct sentences appear redundant')
    print(f'- Total {sentences_to_remove}/{len(df_sentences)} will be removed to avoid have redundancy')
    print(f'- Total {len(sentence_dict)} distinctly generated sentences')

    return df


def sort_sentence_collection_according_to_sentence_length(sentence_collection):
    def get_key(obj):
        return len(obj.sentence)

    sorted_collection = sorted(sentence_collection, key=get_key)
    return sorted_collection


def get_formal_labels(label):
    verb_label_map = {
        "Pres": "Present",
        "Past": "Past"
    }
    sent_label_map = {
        "Pos": "Positive",
        "Neg": "Negative"
    }
    pron_label_map = {
        "Plural": "Plural",
        "Singular": "Singular",
        "Neutral": "Neutral"
    }

    verb = ""
    sentiment = ""
    pronoun = ""
    for key in verb_label_map.keys():
        if key in label:
            verb = verb_label_map[key]
            break
    for key in sent_label_map.keys():
        if key in label:
            sentiment = sent_label_map[key]
            break
    for key in pron_label_map.keys():
        if key in label:
            pronoun = pron_label_map[key]
            break

    return verb, sentiment, pronoun


def prepare_attr_generated_sentences_for_mturk(task, model_a, model_b):
    sentences_A_complete = pd.read_csv(pathlib.Path('..', f'{task}_generated_output', f'{model_a}.txt'))
    sentences_B_complete = pd.read_csv(pathlib.Path('..', f'{task}_attr_generated_output', f'{model_b}.txt'))

    sentences_A = remove_redundant_sentences(sentences_A_complete, model_a)
    sentences_B = remove_redundant_sentences(sentences_B_complete, model_b)

    labels = ["PresPosPlural", "PresPosSingular", "PresPosNeutral",
              "PastPosPlural", "PastPosSingular", "PastPosNeutral",
              "PresNegPlural", "PresNegSingular", "PresNegNeutral",
              "PastNegPlural", "PastNegSingular", "PastNegNeutral"]

    list_of_sentence_pairs = []
    stats = []
    for label in labels:
        if skip_neutral and "Neutral" in label:
            continue
        df_per_label_A = sentences_A.query(f"label == '{label}'")
        df_per_label_B = sentences_B.query(f"label == '{label}'")

        sub_sentence_collection_A, stats_A = create_sentence_objects(df_per_label_A, model_a, label)
        sub_sentence_collection_B, stats_B = create_sentence_objects(df_per_label_B, model_b, label)

        sub_sentence_collection_A = sort_sentence_collection_according_to_sentence_length(sub_sentence_collection_A)
        sub_sentence_collection_B = sort_sentence_collection_according_to_sentence_length(sub_sentence_collection_B)

        stats.append(stats_A)
        stats.append(stats_B)

        number_of_obj = min(len(sub_sentence_collection_A), len(sub_sentence_collection_B))
        for i in range(number_of_obj):
            index = randrange(2)
            dict = {1 - index: sub_sentence_collection_A[i], index: sub_sentence_collection_B[i]}
            # randomly assign objA and objB to index 0 or 1 for each pair created
            sentence_pair = (dict[0], dict[1])
            list_of_sentence_pairs.append(sentence_pair)

    random.shuffle(list_of_sentence_pairs)

    rows = []
    id = 0
    for pair in list_of_sentence_pairs:
        assert pair[0].label == pair[1].label
        verb, sentiment, pronoun = get_formal_labels(pair[0].label)
        row = {'id': id, 'text1': pair[0].sentence, 'text2': pair[1].sentence,
               'model1': pair[0].model_name, 'model2': pair[1].model_name, label: pair[0].label,
               'verb_tense': verb, 'sentiment': sentiment, 'pronoun': pronoun}
        rows.append(row)
        id += 1
    # create dataframes
    df = pd.DataFrame(rows)
    df.set_index('id')
    df_mturk = df.filter(['text1', 'text2', 'model1', 'model2', 'verb_tense', 'sentiment', 'pronoun'], axis=1)
    df_stats = pd.DataFrame(stats)

    # write stats and other metadata in csv files
    # write the shuffled pairs to a .csv file for mturk
    date_folder = time.strftime('%d_%b_%Y_%H:%M:%S_%Z')
    if not os.path.exists(pathlib.Path(mturk_dir, date_folder)):
        os.makedirs(pathlib.Path(mturk_dir, date_folder))
    metadata_file = f'metadata-{model_a}--{model_b}.csv'
    stats_file = pathlib.Path(mturk_dir, date_folder, f'stats-{model_a}--{model_b}.csv')
    source_data_file = pathlib.Path(mturk_dir, date_folder, metadata_file)
    mturk_file = pathlib.Path(mturk_dir, date_folder, f'complete-task-{model_a}--{model_b}.csv')
    df.to_csv(source_data_file, index_label='id')
    df_mturk.to_csv(mturk_file, index_label='id')
    df_stats.to_csv(stats_file, index_label='id')

    # split the data into several chunks with 500 sentences each
    steps = 500
    size = int(len(df_mturk) / steps) * steps
    for i in range(0, size, steps):
        short_mturk_file = pathlib.Path(mturk_dir, date_folder, f'short-task-{i + steps}-{model_a}--{model_b}.csv')
        df_short = df_mturk[i:(i + steps)]
        df_short.to_csv(short_mturk_file, index_label='id')


def prepare_e2e_generated_sentences_for_mturk(task, model_a, model_b):
    sentences_A_complete = pd.read_csv(pathlib.Path('..', f'{task}_generated_output', f'{model_a}.csv'))
    sentences_B_complete = pd.read_csv(pathlib.Path('..', f'{task}_generated_output', f'{model_b}.csv'))

    df = pd.merge(sentences_A_complete, sentences_B_complete, on='MR', suffixes=('_a', '_b'), how='inner')
    list_of_sentence_pairs = []
    for i in range(len(df)):
        # remove identical generated outputs
        if df.output_a.iloc[i] == df.output_b.iloc[i]:
            continue
        sentence_obj_a = Sentence(sentence=df.output_a.iloc[i],
                                  label=df.MR.iloc[i],
                                  model_name=model_a,
                                  clean_text=False)
        sentence_obj_b = Sentence(sentence=df.output_b.iloc[i],
                                  label=df.MR.iloc[i],
                                  model_name=model_b,
                                  clean_text=False)
        index = randrange(2)
        dict = {1 - index: sentence_obj_a, index: sentence_obj_b}
        # randomly assign objA and objB to index 0 or 1 for each pair created
        sentence_pair = (dict[0], dict[1])
        list_of_sentence_pairs.append(sentence_pair)

    random.shuffle(list_of_sentence_pairs)

    rows = []
    for pair in list_of_sentence_pairs:
        assert pair[0].label == pair[1].label
        row = {'text1': pair[0].sentence, 'text2': pair[1].sentence,
               'model1': pair[0].model_name, 'model2': pair[1].model_name,
               'label': pair[0].label}
        rows.append(row)

    # create dataframes
    df = pd.DataFrame(rows)

    # write the shuffled pairs to a .csv file for mturk
    date_folder = time.strftime('%d_%b_%Y_%H:%M:%S_%Z')
    if not os.path.exists(pathlib.Path(mturk_dir, date_folder)):
        os.makedirs(pathlib.Path(mturk_dir, date_folder))

    mturk_file = pathlib.Path(mturk_dir, date_folder, f'complete-task-{model_a}--{model_b}.csv')
    df.to_csv(mturk_file, index_label='id')
    mturk_sample_file = pathlib.Path(mturk_dir, date_folder, f'500-samples-{model_a}--{model_b}.csv')
    df_sample = df.sample(n=500, replace=False)
    df_sample.to_csv(mturk_sample_file, index_label='id')


if __name__ == "__main__":
    # easy mode
    # test_model_a = '2021-Apr-27-09:53:10_6000_1'
    # test_model_b = '2021-Apr-30-09:22:17_6000_1'

    # hard mode
    test_model_a = '2021-Apr-27-09:53:10_6000_1'
    test_model_b = '2021-Apr-26-12:28:06_6000_1'
    skip_neutral = False

    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_a', type=str, default=test_model_a)
    parser.add_argument('--model_b', type=str, default=test_model_b)
    parser.add_argument('--task', type=str, choices=[ATTR_CONTROL, E2E], default=ATTR_CONTROL)
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    model_a = args.model_a
    model_b = args.model_b
    task = args.task

    mturk_dir = 'mturk_data'
    if not os.path.exists(mturk_dir):
        os.makedirs(mturk_dir)

    if task == ATTR_CONTROL:
        prepare_attr_generated_sentences_for_mturk(task, model_a, model_b)
    elif task == E2E:
        prepare_e2e_generated_sentences_for_mturk(task, model_a, model_b)

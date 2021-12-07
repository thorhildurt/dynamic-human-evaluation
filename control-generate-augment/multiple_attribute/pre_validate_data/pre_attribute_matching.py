import pathlib
import pathlib
import numpy as np
import pandas as pd
import spacy
import os
import pronoun_classification as pc
import tense_classification as tc

nlp = spacy.load('en_core_web_sm')

CLASSIFY_PRON = True
CLASSIFY_TENSE = True


def classify_generated_sentences(sentences, file_name):

    # pronoun classification
    if CLASSIFY_PRON:
        pc.labeling(sentences, file_name)
    # tense classification
    if CLASSIFY_TENSE:
        tense_labels = tc.generate_tense_label_files(sentences)
        labels = np.asarray(tense_labels)
        df_labels = pd.DataFrame({'Sentence': sentences[:100],
                                  'Present': labels[:, 0],
                                  'Past': labels[:, 1]})

        result_file = pathlib.Path(result_path, f'{file_name}-tense-pred')
        df_labels.to_csv(result_file, index=False)


def attribute_matching(reviews, labels, file_name):
    num_sentences = len(reviews[:100])

    # Pronoun attribute matching
    pron_pred_results = pd.read_csv(pathlib.Path('results', f'{file_name}-pron-pred'))
    sing_cnt = 0
    pron_cnt = 0
    plural_cnt = 0
    for i in range(num_sentences):
        assert (pron_pred_results.Sentence.iloc[i] == reviews.review.iloc[i])
        if pron_pred_results.Singular.iloc[i] and labels.Singular.iloc[i]:
            sing_cnt += 1
        if pron_pred_results.Neutral.iloc[i] and labels.Neutral.iloc[i]:
            pron_cnt += 1
        if pron_pred_results.Plural.iloc[i] and labels.Plural.iloc[i]:
            plural_cnt += 1
    print(f'Singular-match-count={sing_cnt}, neutral-match-count={pron_cnt}, plural-match-count={plural_cnt}')
    print("Pronoun", (sing_cnt+pron_cnt+plural_cnt) / num_sentences)

    # Tense attribute matching
    tense_pred_results = pd.read_csv(pathlib.Path('results', f'{file_name}-tense-pred'))
    pres_cnt = 0
    past_cnt = 0
    for i in range(num_sentences):
        assert (tense_pred_results.Sentence.iloc[i] == reviews.review.iloc[i])

        if tense_pred_results.Present.iloc[i] and labels.Present.iloc[i]:
            pres_cnt += 1
            continue
        if tense_pred_results.Past.iloc[i] and labels.Past.iloc[i]:
            past_cnt += 1
            continue
    print(f'Present-match-count={pres_cnt}, Past-match-count={past_cnt}')
    print("Tense", (pres_cnt + past_cnt) / num_sentences)


if __name__ == "__main__":
    data_dir = pathlib.Path('..', '..', 'data', 'yelp', 'pron_sentiment_tense')

    file_path_data = os.path.join(data_dir, 'yelp_train.csv')
    file_path_labels = os.path.join(data_dir, 'y_yelp_train.csv')
    result_path = pathlib.Path('results')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # read in the file we want to evaluate via attribute matching
    original_data = pd.read_csv(file_path_data)
    original_labels = pd.read_csv(file_path_labels)

    classify_generated_sentences(original_data['review'], 'yelp_train')
    attribute_matching(original_data, original_labels, 'yelp_train')

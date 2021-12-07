import numpy as np
import pathlib
import pandas as pd
import krippendorff
import shared_functions as c
import os


def fleiss_kappa_manual(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects and 'k' = the number of categories.
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators
    tot_annotations = N * n_annotators  # the total # of annotations
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    print('The distribution of each category over all annotations', p)
    PbarE = np.sum(p * p)  # average chance agreement over all categories
    print('Avg. change of agreements', PbarE)

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items
    print('Observed agreement', Pbar)
    return round((Pbar - PbarE) / (1 - PbarE), 4)


if __name__ == "__main__":
    batch_names = ['Batch_4444974', 'Batch_4447602', 'Batch_4483006', 'Batch_4489733']
    comp = ['GCA vs V1', 'GCA vs V2 (R1)', 'GCA vs V2 (R2)', 'Sheffield2 vs Slug2Slug']
    results = []
    for i, batch in enumerate(batch_names):
        batch_file = f"{batch}_batch_results_post"

        mturk_path = pathlib.Path('..', 'mturk', "results", "production", f'{batch}_batch_results.csv')
        df = pd.read_csv(mturk_path, usecols=["Input.id", "Answer.Selection.label"])
        mturk_data = df.rename(columns={"Input.id": "id", "Answer.Selection.label": "answer"})

        req_ids = mturk_data.id.unique()
        req_annotations = []
        row_req_col_labels = []
        for r_id in req_ids:
            annotations = mturk_data[mturk_data['id'] == r_id]
            answers = annotations[annotations['answer'].isin(['Sentence A', 'Sentence B'])].answer
            row = [1 if i == 'Sentence A' else 2 for i in answers]
            count1 = row.count(1)

            row_req_col_labels.append(row)
            cnt_model1 = len(annotations[annotations['answer'] == 'Sentence A'])
            cnt_model2 = len(annotations[annotations['answer'] == 'Sentence B'])
            req_annotations.append([cnt_model1, cnt_model2])

        matrix = np.array(req_annotations)
        # rows: requests
        # column: labels
        row_req_col_labels = np.array(row_req_col_labels)
        # rows: labels
        # column: requests
        row_labels_col_requests = row_req_col_labels.transpose()

        print(f'Batch: {batch}')
        fleiss_kappa_score1 = fleiss_kappa_manual(matrix)

        krippendorff_score = krippendorff.alpha(row_labels_col_requests)
        print(f'Fleiss Kappa for {batch}: {fleiss_kappa_score1}')

        print(f'krippendorff for {batch}: {krippendorff_score}')
        print()

        result = {
            'batch_name': batch,
            'Fleiss Kappa': round(fleiss_kappa_score1, 2),
            'Krippendorff alpha': round(krippendorff_score, 2),
            'Comparison': comp[i]
        }
        results.append(result)

    # write the results for each iteration run
    folder = pathlib.Path(c.CSV_RESULTS_FOLDER)
    if not os.path.exists(pathlib.Path(folder)):
        os.makedirs(pathlib.Path(folder))
    results_df = pd.DataFrame(results)

    results_df.to_csv(pathlib.Path(folder, 'IAA.csv'), index=False)





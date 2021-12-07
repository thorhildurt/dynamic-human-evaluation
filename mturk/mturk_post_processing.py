import pandas as pd
import pathlib
import os
from tqdm import tqdm

answer_mapper = {"Sentence A": "model1", "Sentence B": "model2"}


def simplify_dataframe(df):
    model_name_mapper = {df.model1.iloc[0]: "model1", df.model2.iloc[0]: "model2"}
    model_id_mapper = [{"model1": df.model1.iloc[0], "model2": df.model2.iloc[0]}]
    rows = []
    print("Simplifying dataframe...")
    for i in tqdm(range(len(df))):
        current_row = df.iloc[i]
        model_id = answer_mapper[current_row.answer]
        if model_id == 'model1':
            model_name = current_row.model1
        elif model_id == 'model2':
            model_name = current_row.model2

        row = {
            "id": current_row.id,
            "workerId": current_row.workerId,
            model_name_mapper[current_row.model1]: current_row.text1,
            model_name_mapper[current_row.model2]: current_row.text2,
            "answer": model_name_mapper[model_name],
            # "pronoun": current_row.pronoun,
            # "verb_tense": current_row.verb_tense,
            # "sentiment": current_row.sentiment
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df_model_mapper = pd.DataFrame(model_id_mapper)
    return df, df_model_mapper


def get_unique_request_rows(df, req_ids):
    requests = []
    for i in req_ids:
        request = df.query(f"id == '{i}'").iloc[0]
        row = {}
        row.update(request)
        requests.append(row)
    df_request = pd.DataFrame(requests)
    df_request = df_request.sort_values(by=['id'])
    return df_request


def fill_in_missing_req(df):
    req_ids = df.id.unique()
    worker_ids = df.workerId.unique()
    df_requests = get_unique_request_rows(df, req_ids)

    print("Filling in missing values...")
    for w_id in tqdm(worker_ids):
        answered_request = df.query(f"workerId == '{w_id}'")
        if len(req_ids) == len(answered_request):
            continue
        answered_request = answered_request.sort_values(by=['id'])
        for index, request in df_requests.iterrows():
            answer = answered_request.query(f"id == '{request.id}'")
            # if worker has not answer this HIT/request, fill in a row with an empty answer for corresponding worker
            if answer.shape[0] == 0:
                row = {
                    "id": request.id,
                    "workerId": w_id,
                    "model1": request.model1,
                    "model2": request.model2,
                    "answer": "no_attempt",
                    # "pronoun": request.pronoun,
                    # "verb_tense": request.verb_tense,
                    # "sentiment": request.sentiment
                }

                df = df.append(row, ignore_index=True)
    df = df.sort_values(by=['id'])
    return df


def sandbox_results(batch):
    metadata = "metadata-2021-Apr-27-09:53:10_6000_1--2021-Apr-30-09:22:17_6000_1"
    result_dir = "development"
    mturk_path = pathlib.Path("results", result_dir, f'{batch}.csv')
    metadata_path = pathlib.Path("results", "mturk_source_files", "sandbox_data", f'{metadata}.csv')
    batch_results = pd.read_csv(mturk_path, usecols=["HITId", "Reward", "AssignmentId", "WorkerId", "AssignmentStatus",
                                                     "WorkTimeInSeconds", "Input.id", "Input.text1", "Input.text2",
                                                     "Input.verb_tense", "Input.sentiment", "Input.pronoun",
                                                     "Answer.Selection.label"])

    average_work_time = batch_results.WorkTimeInSeconds.mean()
    median_work_time = batch_results.WorkTimeInSeconds.median()
    print("Average worktime in seocnds: ", average_work_time)
    print("Median worktime in seocnds: ", median_work_time)
    metadata = pd.read_csv(metadata_path, usecols=["id", "model1", "model2", "text1", "text2"])

    df = pd.merge(batch_results, metadata, left_on=['Input.id'], right_on=['id'], how="inner")
    assert (df["Input.text1"].iloc[0] == df["text1"].iloc[0])

    df = df[["Input.id", "WorkerId", "Input.text1", "Input.text2", "Input.verb_tense", "Input.sentiment",
             "Input.pronoun", "Answer.Selection.label", "model1", "model2"]]

    df = df.rename(columns={"Input.id": "id",
                            "WorkerId": "workerId",
                            "Input.text1": "text1",
                            "Input.text2": "text2",
                            "Input.verb_tense": "verb_tense",
                            "Input.pronoun": "pronoun",
                            "Input.sentiment": "sentiment",
                            "Answer.Selection.label": "answer",
                            })

    df, model_mapper = simplify_dataframe(df)
    df = fill_in_missing_req(df)

    # Write human evaluation from MTurk to human-evaluation project directory
    # for testing label aggregation methods
    project_path = pathlib.Path(__file__).parents[1]
    mturk_data_path = pathlib.Path(project_path, 'human-evaluation', 'mturk_data')
    if not os.path.exists(mturk_data_path):
        os.makedirs(mturk_data_path)
    human_eval_data_file = pathlib.Path(project_path, 'human-evaluation', 'mturk_data', f'{batch}_post.csv')
    model_mapper_file = pathlib.Path(project_path, 'human-evaluation', 'mturk_data', f'{batch}_model_mapper.csv')
    df.to_csv(human_eval_data_file, index=False)
    model_mapper.to_csv(model_mapper_file, index=False)


def production(batch):
    result_dir = "production"
    mturk_path = pathlib.Path("results", result_dir, f'{batch}.csv')

    df = pd.read_csv(mturk_path, usecols=["HITId", "Reward", "AssignmentId", "WorkerId", "AssignmentStatus",
                                                     "WorkTimeInSeconds", "Input.id", "Input.text1", "Input.text2",
                                                     "Input.model1", "Input.model2", "Answer.Selection.label"])

    average_work_time = df.WorkTimeInSeconds.mean()
    median_work_time = df.WorkTimeInSeconds.median()
    total_distinct_workers = len(df.WorkerId.unique())
    print("Average worktime in seconds: ", average_work_time)
    print("Median worktime in seconds: ", median_work_time)
    print("Total distinct workers: ", total_distinct_workers)

    df = df[["Input.id", "WorkerId", "Input.text1", "Input.text2", "Answer.Selection.label", "Input.model1", "Input.model2"]]

    df = df.rename(columns={"Input.id": "id",
                            "WorkerId": "workerId",
                            "Input.text1": "text1",
                            "Input.text2": "text2",
                            "Input.model1": "model1",
                            "Input.model2": "model2",
                            "Answer.Selection.label": "answer",
                            })

    df, model_mapper = simplify_dataframe(df)
    df = fill_in_missing_req(df)

    # Write human evaluation from MTurk to human-evaluation project directory
    # for testing label aggregation methods
    project_path = pathlib.Path(__file__).parents[1]
    mturk_data_path = pathlib.Path(project_path, 'human-evaluation', 'mturk_data')
    if not os.path.exists(mturk_data_path):
        os.makedirs(mturk_data_path)
    human_eval_data_file = pathlib.Path(project_path, 'human-evaluation', 'mturk_data', f'{batch}_post.csv')
    model_mapper_file = pathlib.Path(project_path, 'human-evaluation', 'mturk_data', f'{batch}_model_mapper.csv')
    df.to_csv(human_eval_data_file, index=False)
    model_mapper.to_csv(model_mapper_file, index=False)


if __name__ == "__main__":
    is_dev = False

    if is_dev:
        batch_name = "Batch_319304_batch_results"
        sandbox_results(batch_name)
    else:
        # easy
        # batch_name = "Batch_4444974_batch_results"
        # hard v1
        # batch_name = "Batch_4447602_batch_results"
        # hard v2
        batch_name = "Batch_4483006_batch_results"

        production(batch_name)



import pandas as pd
import pathlib
import plot_one_worker_analysis as pf
import plot_mv_analysis as pmv
import plot_max_three_worker_analysis as ptw
import shared_functions as c
import os
import argparse
from tqdm import tqdm
import numpy as np


def analyse_one_worker_performance(data, batch_name):
    dataframe_rows = []
    workers = data.workerId.unique()
    requests = data.id.unique()
    np.random.seed(c.SEED)
    if c.SEED != 0:
        np.random.shuffle(requests)
    print(analyse_one_worker_performance.__name__)
    for worker_id in tqdm(workers):
        df_requests = data[data['workerId'] == worker_id]

        model1_answers = 0
        model2_answers = 0
        no_attempt = 0
        current_proportion = 0.5

        for i, req_id in enumerate(requests):
            answer = df_requests[df_requests['id'] == req_id]['answer'].iloc[0]

            is_answer = True
            if answer == 'model1':
                model1_answers += 1
            elif answer == 'model2':
                model2_answers += 1

            # if there is no answer for a corresponding request for a given worker
            # then we keep do not update the proportion of answers
            if answer == "no_attempt":
                no_attempt += 1
                is_answer = False
                proportion_of_model1 = current_proportion
            else:
                proportion_of_model1 = model1_answers / (model1_answers + model2_answers)

            row = {
                "reqId": req_id,
                "index": i + 1,
                "workerId": worker_id,
                "is_answer": is_answer,
                "proportion_model1": proportion_of_model1,
                "proportion_model2": 1 - proportion_of_model1
            }
            dataframe_rows.append(row)
            # keep track of the current proportion to fill in
            # proportion for workers that do not attempt specific requests
            current_proportion = proportion_of_model1
        print(f'worker_id: {worker_id}, num_comparisons: {len(df_requests) - no_attempt}')
    df = pd.DataFrame(dataframe_rows)
    df.to_csv(pathlib.Path("dataframes", batch_name, c.ONE_WORKER_FILE_NAME), index=False)


def analyse_one_random_worker_performance(data, batch_name):
    dataframe_rows = []
    requests = data.id.unique()
    np.random.seed(c.SEED)
    if c.SEED != 0:
        np.random.shuffle(requests)
    print(analyse_one_random_worker_performance.__name__)
    for i in tqdm(range(c.ITERATIONS)):
        model1_answers = 0
        model2_answers = 0
        for index, req_id in enumerate(requests):
            df_request_judgements = data.query(f"id == '{req_id}' and answer != 'no_attempt'")
            sample = df_request_judgements.sample(n=1)

            if sample["answer"].iloc[0] == "model1":
                model1_answers += 1
            else:
                model2_answers += 1

            proportion_of_model1 = model1_answers / (index + 1)

            row = {
                "iteration": i,
                "index": index + 1,
                "reqId": req_id,
                "answer": sample["answer"].iloc[0],
                "workerId": sample["workerId"],
                "proportion_model1": proportion_of_model1,
                "proportion_model2": 1 - proportion_of_model1
            }

            dataframe_rows.append(row)
    df = pd.DataFrame(dataframe_rows)
    df.to_csv(pathlib.Path("dataframes", batch_name, c.RANDOM_WORKER_FILE_NAME), index=False)


def analyse_mv_performance(data, batch_name, num_votes=5):
    dataframe_rows = []
    requests = data.id.unique()
    np.random.seed(c.SEED)
    if c.SEED != 0:
        np.random.shuffle(requests)
    print(analyse_mv_performance.__name__)
    for i in tqdm(range(c.ITERATIONS)):
        model1_answers = 0
        model2_answers = 0
        for index, req_id in enumerate(requests):
            df_request_judgements = data.query(f"id == '{req_id}' and answer != 'no_attempt'")
            sample = df_request_judgements.sample(n=num_votes)
            model1_votes = sample[sample["answer"] == "model1"]
            ratio = len(model1_votes) / num_votes
            if ratio > 0.5:
                model1_answers += 1
            else:
                model2_answers += 1

            proportion_of_model1 = model1_answers / (index + 1)
            row = {
                "iteration": i,
                "reqId": req_id,
                "answer": sample["answer"].iloc[0],
                "index": index + 1,
                "workerIds": list(sample["workerId"]),
                "proportion_model1": proportion_of_model1,
                "proportion_model2": 1 - proportion_of_model1
            }
            dataframe_rows.append(row)

    df = pd.DataFrame(dataframe_rows)
    df.to_csv(pathlib.Path("dataframes", batch_name, c.MV_WORKER_FILE_NAME.format(num_votes)), index=False)


def analyse_max_three_workers_performance(data, batch_name):
    dataframe_rows = []
    requests = data.id.unique()
    np.random.seed(c.SEED)
    if c.SEED != 0:
        np.random.shuffle(requests)
    print(analyse_max_three_workers_performance.__name__)
    for i in tqdm(range(c.ITERATIONS)):
        model1_answers = 0
        model2_answers = 0
        work_counter = 0
        for index, req_id in enumerate(requests):
            df_request_judgements = data.query(f"id == '{req_id}' and answer != 'no_attempt'")
            sample = df_request_judgements.sample(n=3)
            two_judges = sample.iloc[:2]
            model1_votes = two_judges[two_judges["answer"] == "model1"]
            if len(model1_votes) != 2:
                ratio = len(sample[sample["answer"] == "model1"]) / 3
                if ratio > 0.5:
                    model1_answers += 1
                else:
                    model2_answers += 1
                work_counter += 3
            elif len(model1_votes) == 2:
                model1_answers += 1
                work_counter += 2
            else:
                model2_answers += 1
                work_counter += 2

            proportion_of_model1 = model1_answers / (index + 1)

            row = {
                "iteration": i,
                "reqId": req_id,
                "answer": sample["answer"].iloc[0],
                "index": index + 1,
                "total_work": work_counter,
                "workerIds": list(sample["workerId"]),
                "proportion_model1": proportion_of_model1,
                "proportion_model2": 1 - proportion_of_model1
            }

            dataframe_rows.append(row)

    df = pd.DataFrame(dataframe_rows)
    df.to_csv(pathlib.Path("dataframes", batch_name, c.MAX_THREE_WORKER_FILE_NAME), index=False)


def main(args):
    env = args.env
    batch_name = args.batch_name

    # specify which model id we want to plot in our experiments
    y_axis = args.y_axis
    include_worker_analysis = args.worker_analysis
    result_file = "two_choice_mturk_results.csv"

    model_names_file = 'model_names'
    batch_file = f"{batch_name}_batch_results_post"
    model_mapper_file = f"{batch_name}_batch_results_model_mapper"
    path = pathlib.Path('mturk_data', f'{batch_file}.csv')
    data = pd.read_csv(path)
    path = pathlib.Path('mturk_data', f'{model_mapper_file}.csv')
    models = pd.read_csv(path)
    path = pathlib.Path('mturk_data', f'{model_names_file}.csv')
    model_names = pd.read_csv(path)
    label_name1 = model_names[model_names['model_id'] == models.model1[0]].iloc[0].plot_name
    label_name2 = model_names[model_names['model_id'] == models.model2[0]].iloc[0].plot_name
    Model1 = c.Model(model_id="model1", model_name=models.model1[0], label_name=label_name1,
                     model_column="proportion_model1", nlg_task=args.task, criteria=args.criteria)
    Model2 = c.Model(model_id="model2", model_name=models.model2[0], label_name=label_name2,
                     model_column="proportion_model2", nlg_task=args.task, criteria=args.criteria)
    print(f'Running {batch_name}...')
    print(f'Target delta: {c.DELTA}')
    print(f'Model1 = {Model1.model_name}')
    print(f'Model2 = {Model2.model_name}')
    mapper = {"model1": Model1, "model2": Model2}

    # store the batch name with a SEED representing the request pair ordering
    if c.SEED != 0:
        batch_name = f'{batch_name}_seed_{c.SEED}'
    path = pathlib.Path("dataframes", batch_name)
    if not os.path.exists(path):
        os.makedirs(path)

    if c.MV:
        print("Running mv worker analysis...")
        # write dataframe to csv with mv results
        # set CREATE_DF to True to write/update the dataframe for mv
        if c.CREATE_DF:
            analyse_mv_performance(data, batch_name, num_votes=5)
            analyse_mv_performance(data, batch_name, num_votes=7)

        pmv.plot_mv_with_ci(env=env, batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, result_file=result_file,
                            num_votes=5, color="C0")
        pmv.plot_mv_with_ci(env=env, batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, result_file=result_file,
                            num_votes=7, color="C1")

        pmv.plot_mv_labelling_effort(batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, num_votes=7, color="C0")
        pmv.plot_mv_labelling_effort(batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, num_votes=5, color="C1")

    if c.Max_THREE:
        print("Running max three worker analysis...")
        # write dataframe to csv with max three worker results
        # set CREATE_DF to True to write/update the dataframe for mv
        if c.CREATE_DF:
            analyse_max_three_workers_performance(data, batch_name)
        ptw.plot_max_three_with_ci(env=env, batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, result_file=result_file)
        ptw.plot_max_three_labelling_effort(batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, color="C2")

    if c.ONE_WORKER:
        print("Running one/random worker analysis...")
        # write dataframes to csv with one worker results
        # set CREATE_DF to True to write/update the dataframe for one worker
        if c.CREATE_DF:
            analyse_one_random_worker_performance(data, batch_name)
            if include_worker_analysis:
                analyse_one_worker_performance(data, batch_name)

        one_worker_path = pathlib.Path("dataframes", batch_name, c.ONE_WORKER_FILE_NAME)
        if include_worker_analysis:
            # plot the proportion of answers towards one model for each worker with ci
            pf.plot_proportion_per_worker_with_ci(env=env, batch_name=batch_name, path=one_worker_path, y_axis=y_axis,
                                                  model_mapper=mapper)
        if include_worker_analysis and env == "dev":
            # only plots for workers that attempted all HITs which is possible to analyse with collected sandbox data
            pf.plot_proportion_per_workers_with_all_answers(env=env, batch_name=batch_name, path=one_worker_path, y_axis=y_axis,
                                                            model_mapper=mapper)
            # only plots for workers that attempted all HITs which is possible to analyse with collected sandbox data
            pf.plot_mean_proportion_for_workers_with_all_answers_with_ci(env=env, batch_name=batch_name, path=one_worker_path,
                                                                         y_axis=y_axis, model_mapper=mapper)

        random_worker_path = pathlib.Path("dataframes", batch_name, c.RANDOM_WORKER_FILE_NAME)
        # plot the mean proportion when one random worker answers each HIT
        pf.plot_mean_proportion_for_random_worker_with_ci(env=env, batch_name=batch_name, path=random_worker_path,
                                                          y_axis=y_axis, model_mapper=mapper, result_file=result_file)
        pf.plot_random_worker_labelling_effort(batch_name=batch_name, y_axis=y_axis, model_mapper=mapper, color="C4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="dev")
    parser.add_argument('--batch_name', type=str, default="Batch_319304")
    parser.add_argument('--y_axis', type=str, default="model1")
    parser.add_argument('--worker_analysis', type=int, default=0)
    parser.add_argument('--criteria', type=str, default="naturalness")
    parser.add_argument('--task', type=str, default="CGA")
    arguments = parser.parse_args()
    main(arguments)

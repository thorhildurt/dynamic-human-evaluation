import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from enum import Enum
from sklearn.utils import resample
''''
BEGIN GLOBAL CONFIG VARIABLES
'''

ONE_WORKER = True
Max_THREE = True
MV = True

ITERATIONS = 100
# Only applied using human eval. data collected via RankME
DATAFRAMES = 10
DELTA = 0.01
PROBABILITY = 1.0 - DELTA
SAVE = True
CREATE_DF = True
ALPHA_STD = .4
# sets the ordering of accumulated request pairs
# set to 0 for the original order on mturk
SEED = 0
# sets the random decision when there is a tie (used for analysing rankme data)
TIE_SEED = 10
''''
END GLOBAL CONFIG VARIABLES
'''

ONE_WORKER_FILE_NAME = f"one_worker_proportion.csv"
RANDOM_WORKER_FILE_NAME = f"random_worker_proportion_{ITERATIONS}_iter.csv"
MV_WORKER_FILE_NAME = "mv_with_{0}_voters_" + str(ITERATIONS) + "_iter.csv"
MAX_THREE_WORKER_FILE_NAME = f"max_three_worker_proportion_{ITERATIONS}_iter.csv"

CSV_RESULTS_FOLDER = 'csv_results'
PLOT_FOLDER = 'visual_data_analysis'

RANDOM_WORKER_NAME = "Random worker evaluation"
MV_WORKER_NAME = "MV={0} worker evaluation"
MAX_THREE_WORKER_NAME = "Max three worker evaluation"

# short version of the method name
MAX_THREE_WORKER_LABEL = "Max 3 Workers"
MV_WORKER_LABEL = "{0} Workers"
RANDOM_WORKER_LABEL = 'Random Worker'

# ids to append to file names
MV_ID = "MV_{0}"
MAX_THREE_ID = "MAX3"
R_ID = "ONE_R"


class Model:
    def __init__(self, model_id, model_name, label_name, model_column, nlg_task, criteria):
        self.model_id = model_id
        self.model_name = model_name
        self.label_name = label_name
        self.model_column = model_column
        self.nlg_task = nlg_task
        self.criteria = criteria


class DecisionBound(Enum):
    LOWER_BOUND = 1
    UPPER_BOUND = 2
    NO_DECISION = 3


def get_ci(x, delta):
    sample_sizes = x[x >= 1]
    ci = np.sqrt((-1 * np.log(delta)) / (2 * sample_sizes))
    ci = np.insert(ci, 0, np.inf)
    return ci


def _find_max_ci(x, y):
    sample_sizes = x
    t = np.array([y[i] - 0.5 if y[i] >= 0.5 else 0.5 - y[i] for i, n in enumerate(sample_sizes)])
    deltas = np.exp(-(t ** 2) * 2 * sample_sizes)
    return sample_sizes, deltas


def delta_analysis_v2(env, name, batch_name, data, proportion_column, color):
    # directory for the results
    folder = pathlib.Path("max_confidence_intervals")
    if not os.path.exists(pathlib.Path(PLOT_FOLDER, env, batch_name, folder)):
        os.makedirs(pathlib.Path(PLOT_FOLDER, env, batch_name, folder))

    iterations = data.iteration.unique()
    req_ids = data.reqId.unique()
    x = np.arange(len(req_ids))

    total_max_probabilities = []
    for i in iterations:
        df_method_run = data[data['iteration']==i]
        df_method_run = df_method_run.reset_index(drop=True)
        proportions = df_method_run[proportion_column]
        sample_sizes, deltas = _find_max_ci(x, proportions)
        max_probabilities = np.ones(len(deltas)) - deltas
        total_max_probabilities.append(max_probabilities)

    fig, ax = plt.subplots(figsize=(4, 3))

    avg_prob = np.mean(total_max_probabilities, axis=0)
    std_prob = np.std(total_max_probabilities, axis=0)

    filtered_prob = np.where(np.array(avg_prob) > (1.0 - DELTA))[0]
    threshold_for_target_delta = -1 if filtered_prob.size == 0 else filtered_prob[0] + 1
    if threshold_for_target_delta >= 0:
        delta_str = r'($1-\delta$)'
        ax.axvline(x=threshold_for_target_delta, ymin=0, ymax=1,
                   label=f'Requests={threshold_for_target_delta}, {delta_str}={1 - DELTA}',
                   linestyle='dotted', color=color)
        ax.legend()

    ax.set_ylabel(r'Probability ($1-\delta$)')
    ax.set_xlabel('Number of Requests')
    # plot the expected mean for each n
    ax.plot(x, avg_prob, color=color)
    ax.fill_between(x, avg_prob - std_prob, np.minimum(np.ones(len(x)), avg_prob + std_prob), color=color, alpha=.25)
    plt.ylim([0, 1.001])
    plt.xlim([0, len(x)])
    save_plot(env, fig, batch_name, pathlib.Path(folder, f"{name}_max_CI_probability"))


def get_decision(x, y, ci):
    lower_ci = y - ci
    upper_ci = y + ci
    decision = -1
    index = 0
    decision_bound = DecisionBound.NO_DECISION
    for idx, val in enumerate(lower_ci):
        if val > 0.5 and y[idx] > 0.5 and decision < 0:
            decision = x[idx]
            index = idx - 1
            decision_bound = DecisionBound.LOWER_BOUND
        if val < 0.5:
            decision = -1
            decision_bound = DecisionBound.NO_DECISION

    if decision_bound == DecisionBound.LOWER_BOUND:
        return decision, index, decision_bound

    for idx, val in enumerate(upper_ci):
        if val < 0.5 and y[idx] < 0.5 and decision < 0:
            decision = x[idx]
            index = idx - 1
            decision_bound = DecisionBound.UPPER_BOUND
        if val > 0.5:
            decision = -1
            decision_bound = DecisionBound.NO_DECISION
    return decision, index, decision_bound


def get_better_model(y_axis, decision_bound):
    better_model_id = ""
    if y_axis == "model1" and decision_bound == DecisionBound.LOWER_BOUND:
        better_model_id = "model1"
    elif y_axis == "model2" and decision_bound == DecisionBound.LOWER_BOUND:
        better_model_id = "model2"
    elif y_axis == "model1" and decision_bound == DecisionBound.UPPER_BOUND:
        better_model_id = "model2"
    elif y_axis == "model2" and decision_bound == DecisionBound.UPPER_BOUND:
        better_model_id = "model1"
    return better_model_id


def save_result(env, file_name, decision, model_mapper, better_model, num_samples, avg_labels, labelling_method):
    is_decision = decision > 0
    t = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    new_entry = {
        'timestamp': t,
        'env': env,
        'is_decision': is_decision,
        'ci_probability': PROBABILITY,
        'model1': model_mapper['model1'].model_name,
        'model2': model_mapper['model2'].model_name,
        'better_model': better_model,
        'num_samples': num_samples,
        'avg_label': avg_labels,
        'labelling_method': labelling_method,
        'criteria': model_mapper['model1'].criteria,
        'task': model_mapper['model1'].nlg_task,
        'iterations': ITERATIONS,
        'tie_seed': TIE_SEED
    }

    df_new_entry = pd.DataFrame(new_entry, index=[0])
    df_new_entry['ci_probability'].round(decimals=3)

    if not os.path.exists(CSV_RESULTS_FOLDER):
        os.makedirs(CSV_RESULTS_FOLDER)
    file_path = pathlib.Path(CSV_RESULTS_FOLDER, file_name)

    if not os.path.isfile(file_path):
        df_new_entry.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, df_new_entry], sort=False)
        df = df.set_index('timestamp')
        df.to_csv(file_path)


def save_plot(env, fig, batch_name, name):
    if SAVE:
        folder = PLOT_FOLDER
        if not os.path.exists(folder):
            os.makedirs(pathlib.Path(folder, env))
        batch_folder = pathlib.Path(folder, env, batch_name)
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)
        img_path = pathlib.Path(batch_folder, f'{name}.pdf')
        fig.tight_layout()
        fig.savefig(img_path, dpi=100, format='pdf')
    else:
        plt.show()

    plt.close()


def update_csv_results_file(file_name, data_entry):
    if not os.path.exists(CSV_RESULTS_FOLDER):
        os.makedirs(CSV_RESULTS_FOLDER)
    file_path = pathlib.Path(CSV_RESULTS_FOLDER, file_name)
    if not os.path.isfile(file_path):
        data_entry.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        df = pd.concat([df, data_entry], sort=False)
        df = df.set_index('timestamp')
        df.to_csv(file_path)


def plot_random_probability_for_requests(x, y):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel(f'Probability of random answer')
    ax.set_xlabel('Request ids')
    ax.plot(x, y, 'go')
    plt.show()


def bootstrap_resample_ci(sample, bootsrap_iterations=1000, confidence=0.99):
    # plot CI with bootstrapping
    bootstrap_iterations = bootsrap_iterations
    n_size = len(sample)
    differences = []
    sample_mean = np.mean(sample)
    for iteration in range(bootstrap_iterations):
        new_sample = resample(sample, n_samples=n_size)
        mean = np.mean(new_sample)
        differences.append(sample_mean - mean)
    differences.sort()
    # confidence intervals
    alpha = confidence
    p = ((1.0 - alpha) / 2.0) * 100
    lower = np.percentile(differences, p)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = np.percentile(differences, p)
    return lower, upper


def update_labelling_effort(batch_name, results_object):

    folder = pathlib.Path(CSV_RESULTS_FOLDER, batch_name)
    if not os.path.exists(pathlib.Path(folder)):
        os.makedirs(pathlib.Path(folder))

    file_path = pathlib.Path(folder, f'avg_label_effort_results_{DELTA}.csv')

    if not os.path.isfile(file_path):
        df_new_entry = pd.DataFrame(results_object, index=[0])
        df_new_entry.to_csv(file_path, index=False)
    else:
        df = pd.read_csv(file_path)
        filter_df = df[df['method_id'] == results_object['method_id']]
        if len(filter_df) == 0:
            df_new_entry = pd.DataFrame(results_object, index=[len(df)])
            df = pd.concat([df, df_new_entry], sort=False)
            df.to_csv(file_path, index=False)
        else:
            df_new_entry = pd.DataFrame(results_object, index=[len(df)])
            idx = df[df['method_id'] == results_object['method_id']].index
            for col in df_new_entry.columns:
                df.at[idx, col] = df_new_entry.iloc[0][col]
            df.to_csv(file_path, index=False)

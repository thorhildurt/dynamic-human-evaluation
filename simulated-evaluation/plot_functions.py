import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pathlib
from textwrap import wrap
from tqdm import tqdm
from sklearn.utils import resample
import time

SAVE = True
PLOT_FOLDER = "visual_analysis"

plt.rc('font', family='serif')

method_short_name_mapper = {
    'MV_7': '7 Workers',
    'MV_5': '5 Workers',
    'MAX3': 'Max 3 Workers',
    'ONE': 'Fixed Worker',
    'ONE_R': 'One Worker'
}

color_map = {
    "MV_5": "C0",
    "MV_7": "C1",
    "MAX3": "C2",
    "ONE": "C3",
    "ONE_R": "C4"
}


def save_plot(folder, fig, name):
    if SAVE:
        if not os.path.exists(pathlib.Path(PLOT_FOLDER, folder)):
            os.makedirs(pathlib.Path(PLOT_FOLDER, folder))
        img_path = pathlib.Path(PLOT_FOLDER, folder, f'{name}.pdf')
        fig.tight_layout()
        fig.savefig(img_path, dpi=100, format='pdf')
    else:
        plt.show()

    plt.close()


def get_threshold(x, y, ci):
    lower_ci = y - ci
    decision_lower = -1

    for idx, val in enumerate(lower_ci):
        if val > 0.5 and y[idx] > 0.5 and decision_lower < 0:
            decision_lower = x[idx]
        if val < 0.5:
            decision_lower = -1

    return decision_lower


def _get_hoeffding_bound(x, delta):
    return np.sqrt((-1 * np.log(delta)) / (2 * x))


def plot_proportion_per_request_for_a_single_method_with_ci(test_id, folder, df, rd_mean, var, method_id, color, delta,
                                                            procedure, num_requests, n_iterations):
    start = time.time()
    print("plot_proportion_per_request_for_a_single_method_with_ci:", method_id)
    name = f"test_id={test_id}_proportion_per_request_method={method_id}_mean={rd_mean}_eval_method={procedure}"

    df = df[df["rd_mean"] == rd_mean]
    df_method = df[df["evaluation_id"] == method_id]

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.set_ylabel('Mean Proportion for Model A')
    ax.set_xlabel('Number of Request Pairs')

    plt.axhline(0.5, color='gray')
    plt.title(method_short_name_mapper[method_id])

    # estimate the mean of the unknown distribution post Bernoulli trial
    means = df_method.groupby('req_id').mean().proportion_of_1
    # convert dask df to pandas df
    means = means.compute()

    x1 = means.index.values[:num_requests]
    y1 = means.values[:num_requests]

    # we compute ci using Hoeffding’s inequality, with given sample size
    # and probability of error to compute the error tolerance (ci)
    ci = _get_hoeffding_bound(x1, delta)

    # plot the ci with given probability
    ax.fill_between(x1, np.maximum(np.zeros(len(x1)), (y1 - ci)),
                    np.minimum(np.ones(len(x1)), (y1 + ci)), color=color, alpha=.1)

    # find the threshold when our lower bound is strictly larger than 0.5
    decision = get_threshold(x1, y1, ci)

    if decision >= 0:
        ax.axvline(x=decision, ymin=0, ymax=1, linestyle='--', color=color, label=f'Req={decision}')
        ax.legend()

    # plot the expected mean for each n
    plt.xlim([0, max(x1)])
    plt.ylim([0, 1])
    ax.plot(x1, y1, color=color)

    save_plot(folder, fig, name)
    end = time.time()
    print("Time: ", round((end - start), 2), "sec")


def plot_proportion_per_request_for_all_methods(test_id, folder, df, rd_mean, var, delta, eval_method, num_requests, n_iterations):
    start = time.time()
    print("plot_proportion_per_request_for_all_methods")
    name = f"test_id={test_id}_proportion_per_request_all_methods_mean={rd_mean}_eval_method={eval_method}"

    fig, ax = plt.subplots(figsize=(4, 3))
    df = df[df["rd_mean"] == rd_mean]
    df = df.compute()

    method_ids = ['MV_7', 'MV_5', 'MAX3', 'ONE', 'ONE_R']

    ax.set_ylabel('Mean Proportion for Model A')
    ax.set_xlabel('Number of Request Pairs')
    plt.axhline(0.5, color='gray')

    for i, method in enumerate(method_ids):
        color = color_map[method]
        df_method = df[df["evaluation_id"] == method]
        label = method_short_name_mapper[method]
        # this would be the estimated means
        means = df_method.groupby('req_id').mean().proportion_of_1

        x1 = means.index.values[:num_requests]
        y1 = means.values[:num_requests]

        ci = _get_hoeffding_bound(x1, delta)

        # find the threshold when our ci lower bound is strictly larger than 0.5
        decision = get_threshold(x1, y1, ci)

        project_ids = np.arange(n_iterations)
        labels_per_sample_for_project = []
        for pid in project_ids:
            df_method_run = df_method.query(f"project_id == {pid}")
            df_method_run = df_method_run.reset_index(drop=True)
            label_sum = 0
            labels_per_sample = []
            for idx, val in enumerate(df_method_run.req_id):
                # count the number of labels annotated so far for each sample
                label_sum += df_method_run.labels_per_request[idx]
                labels_per_sample.append(label_sum)
            labels_per_sample_for_project.append(labels_per_sample)
        np_matrix = np.array(labels_per_sample_for_project)
        avg_label_over_samples = -1
        if decision >= 0:
            avg_label_over_samples = np.mean(np_matrix, axis=0)[decision - 1]
            # plot the threshold line
            ax.axvline(x=decision, ymin=0, ymax=1,
                       linestyle='--', color=color)
            threshold_label = f'Avg. Labels={int(avg_label_over_samples)}'
        else:
            threshold_label = ''

        # plot the expected mean for each n
        ax.plot(x1, y1, label=f'{label} ({threshold_label})', color=color)

    plt.xlim([0, max(x1)])

    ax.legend()
    save_plot(folder, fig, name)
    end = time.time()
    print("Time: ", round((end - start), 2), "sec")


# plot proportion per label effort and return average labelling effort in csv format
def plot_proportion_per_label_effort_for_all_methods(test_id, folder, df, rd_mean, delta, eval_method, num_requests, n_iterations):
    start = time.time()
    print("plot_proportion_per_label_effort_for_all_methods")
    name = f"test_id={test_id}_proportion_per_effort_all_methods_mean={rd_mean}_eval_method={eval_method}"

    fig, ax = plt.subplots(figsize=(4, 3))
    df = df[df['rd_mean'] == rd_mean]

    method_ids = ['MV_7', 'MV_5', 'MAX3', 'ONE', 'ONE_R']

    ax.set_ylabel('Mean Proportion for Model A')
    ax.set_xlabel('Labelling Effort')
    plt.axhline(0.5, color='gray')
    labelling_effort_res = []
    decision_miss = False
    count_miss = {}

    for i, method in tqdm(enumerate(method_ids)):
        count_miss[method] = 0
        color = color_map[method]
        df_method = df[df['evaluation_id'] == method]
        df_method_computed = df_method.compute()
        label = method_short_name_mapper[method]

        # this would be the estimated means
        means = df_method.groupby('req_id').mean().proportion_of_1
        # convert dask df to pandas df
        means = means.compute()

        x1 = means.index.values[:num_requests]
        y1 = means.values[:num_requests]
        ci = _get_hoeffding_bound(x1, delta)

        project_ids = np.arange(n_iterations)
        effort_for_decision_per_iteration = []
        requests_for_decision_per_iteration = []
        labelling_efforts_per_req = []
        for pid in project_ids:
            df_iteration = df_method_computed[df_method_computed['project_id'] == pid][:num_requests]
            df_iteration = df_iteration.reset_index(drop=True)
            selection_ratio = df_iteration['proportion_of_1'].tolist()
            decision = get_threshold(x1, selection_ratio, ci)
            if decision < 0:
                labelling_effort = df_iteration['labels_per_request'].sum()
                decision_miss = True
                count_miss[method] += 1
                n = decision
            else:
                labelling_effort = df_iteration['labels_per_request'].iloc[:decision].sum()
                n = decision - 1

            effort_for_decision_per_iteration.append(labelling_effort)
            requests_for_decision_per_iteration.append(df_iteration['req_id'].iloc[n])
            label_sum = 0
            labels_per_req = []
            for idx, val in enumerate(df_iteration.req_id):
                # count the number of labels annotated so far for each sample
                label_sum += df_iteration.labels_per_request[idx]
                labels_per_req.append(label_sum)
            labelling_efforts_per_req.append(labels_per_req)

        mean_label_effort_per_request = np.mean(labelling_efforts_per_req, axis=0)

        average_req_cnt = np.mean(requests_for_decision_per_iteration)
        std_req_cnt = np.std(requests_for_decision_per_iteration)

        average_effort = np.mean(effort_for_decision_per_iteration)
        std_effort = np.std(effort_for_decision_per_iteration)

        # compute confidence intervals with bootstrapping
        sample_mean = np.mean(effort_for_decision_per_iteration)
        lower, upper = _bootstrap_resample_ci(effort_for_decision_per_iteration)
        # sample mean - the lowest difference
        ci_upper = sample_mean - lower
        # sample mean - the highest difference
        ci_lower = sample_mean - upper

        # plot the threshold line
        ax.axvline(x=average_effort, ymin=0, ymax=1,
                   linestyle='--', color=color)
        ax.axvspan(ci_lower, ci_upper, alpha=.25, color=color_map[method])
        threshold_label = f'Avg. Labels={int(average_effort)}'

        # plot the expected mean for each amount of labelling effort
        ax.plot(mean_label_effort_per_request, y1, label=f'{label} ({threshold_label})', color=color)
        results_entry = {
            'test_id': test_id,
            'method': label,
            'rd_mean': rd_mean,
            'labelling_std': std_effort,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci': f'{round(ci_lower)}-{round(ci_upper)}',
            'avg_std': f'${round(average_effort)} \pm {round(std_effort)}$',
            'average_labelling': average_effort,
            'delta': delta,
            'average_requests': average_req_cnt,
            'requests_std': std_req_cnt
        }
        labelling_effort_res.append(results_entry)

    plt.xlim([0, max(x1)])

    ax.legend()
    save_plot(folder, fig, name)

    # save average labelling effort results into csv file
    df = pd.DataFrame(labelling_effort_res)
    raw_data = pathlib.Path("raw_data_results")
    folder = pathlib.Path(folder, f'delta={delta}')
    if not os.path.exists(pathlib.Path(raw_data, folder)):
        os.makedirs(pathlib.Path(raw_data, folder))
    df.to_csv(pathlib.Path(raw_data, folder, f'average_label_{rd_mean}.csv'), index=False)

    if decision_miss is True:
        print("Not all method could reach decision for given amount of requests")
        for i, method in enumerate(method_ids):
            print(f'{rd_mean} - {method}: {count_miss[method]}')

    end = time.time()
    print("Time: ", round((end - start), 2), "sec")


# plot histograms for each method and density curve of corresponding distribution
def plot_average_label_effort_for_decision(test_id, folder, df, rd_mean, delta, eval_method, n_iterations):
    start = time.time()
    print("plot_average_label_effort_for_decision")
    method_ids = ['MV_7', 'MV_5', 'MAX3', 'ONE', 'ONE_R']
    label_effort_collection = {}
    label_collection = {}

    mean_efforts = []
    std_efforts = []
    short_labels = []

    decision_miss = False
    count_miss = {}

    # data processing
    for i, method in tqdm(enumerate(method_ids)):
        count_miss[method] = 0
        # record labelling efforts for each iteration for each method
        labeling_efforts = []
        df_method = df[df['evaluation_id'] == method]
        df_method = df_method[df_method['rd_mean'] == rd_mean]
        df_method = df_method.compute()
        label = method_short_name_mapper[method]
        label_collection[method] = label
        x = df_method.req_id.unique()
        ci = _get_hoeffding_bound(x, delta)

        project_ids = np.arange(n_iterations)
        for i in range(len(project_ids)):
            project_df = df_method[df_method['project_id'] == i]
            project_df = project_df.sort_values(by='req_id')
            prop_over_req = project_df['proportion_of_1']
            # find the threshold when our ci lower bound is strictly larger than 0.5
            decision = get_threshold(x, list(prop_over_req), ci)
            if decision < 0:
                labeling_effort = project_df['labels_per_request'].sum()
                decision_miss = True
                count_miss[method] += 1
            else:
                labeling_effort = project_df['labels_per_request'].iloc[:decision].sum()
            labeling_efforts.append(labeling_effort)

        label_effort_collection[method] = labeling_efforts
        mean = np.mean(labeling_efforts)
        median = np.median(labeling_efforts)

        # plotting configurations
        fig, ax = plt.subplots(figsize=(4, 3))
        plt.title(label)
        ax.set_ylabel('Probability Density')
        ax.set_xlabel('Labelling Effort')
        hist_name = f"test_id={test_id}_avg_labels_hist={rd_mean}_method={method}_eval_method={eval_method}"
        count, bins, _ = ax.hist(labeling_efforts, 10, density=True,
                                 facecolor=color_map[method], edgecolor=color_map[method],  alpha=.5)

        short_labels.append(label)
        mean_efforts.append(mean)
        std_efforts.append(np.std(labeling_efforts))

        save_plot(pathlib.Path(folder, 'histograms'), fig, hist_name)

    # plot density plots w/ seaborn kde for all methods of the labelling efforts for given mean
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Labelling Effort')
    for key in label_effort_collection.keys():
        sns.kdeplot(np.array(label_effort_collection[key]), color=color_map[key],
                    label=label_collection[key], linewidth=2)

    ax.legend()
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    density_plot = f"test_id={test_id}_labels_density={rd_mean}_eval_method={eval_method}"
    save_plot(pathlib.Path(folder, 'histograms'), fig, density_plot)

    if decision_miss is True:
        print("Not all method could reach decision for given amount of requests")
        for i, method in enumerate(method_ids):
            print(f'{rd_mean} - {method}: {count_miss[method]}')

    end = time.time()
    print("Time: ", round((end - start), 2), "sec")


def _bootstrap_resample_ci(sample, bootsrap_iterations=1000, confidence=0.99):
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


# specific plot for the paper
def plot_difficulty_comparison(test_id, folder, df, method_id, delta, procedure, num_requests):
    start = time.time()
    print("plot_difficulty_comparison")
    name = f"test_id={test_id}_compare_difficulties={method_id}_eval_method={procedure}"

    df = df[df['evaluation_id'] == method_id]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylabel('Mean Proportion for Model A')
    ax.set_xlabel('Number of Request Pairs')
    plt.axhline(0.5, color='gray')

    # plot corresponding difficulties
    c_means = [0.125, 0.0625]
    colors = ['#ff9613', '#0f9a99']
    for indx, mean in enumerate(c_means):
        df_filter = df[df['rd_mean'] == mean]
        means = df_filter.groupby('req_id').mean().proportion_of_1
        # convert dask df to pandas df
        means = means.compute()
        x = means.index.values[:num_requests]
        y = means.values[:num_requests]

        # we compute ci using Hoeffding’s inequality, with given sample size
        # and probability of error to compute the error tolerance (ci)
        ci = _get_hoeffding_bound(x, delta)

        # plot the ci with given probability
        ax.fill_between(x, np.maximum(np.zeros(len(x)), (y - ci)),
                       np.minimum(np.ones(len(x)), (y + ci)), color=colors[indx], alpha=.25)

        # find the threshold when our lower bound is strictly larger than 0.5
        decision = get_threshold(x, y, ci)
        if decision >= 0:
            # plot the threshold line
            ax.axvline(x=decision, ymin=0, ymax=1, linestyle='--', color=colors[indx], label=f'$\mu={mean}$')
            ax.legend(loc=1)

        plt.xlim([0, 5000])
        plt.ylim([0.2, 0.8])
        plt.xlim()
        ax.plot(x, y, color=colors[indx])

    save_plot(folder, fig, name)
    end = time.time()
    print("Time: ", round((end - start), 2), "sec")
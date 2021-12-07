import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import shared_functions as c
import os
import math


def plot_max_three_with_ci(env, batch_name, y_axis, model_mapper, result_file, is_rankme=False):
    model_key = "model1"
    if y_axis == "model2":
        model_key = "model2"
    model_id = model_mapper[model_key].model_id
    proportion_column = model_mapper[model_key].model_column

    data = pd.read_csv(pathlib.Path("dataframes", batch_name, c.MAX_THREE_WORKER_FILE_NAME))
    fig, ax = plt.subplots(figsize=(4, 3))

    plt.axhline(0.5, color='gray')
    ax.set_ylabel(f'Mean Proportion for Model {model_mapper[model_key].label_name}')
    ax.set_xlabel('Number of Request Pairs')
    color = "C2"
    if is_rankme:
        proportion_means_per_df = data.groupby(["df_id", "index"]).mean()
        proportion_means = proportion_means_per_df.groupby("index").mean()[proportion_column]
        avg_total_work_per_df = data.groupby(["df_id", "index"]).mean()
        average_total_work = avg_total_work_per_df.groupby("index").mean().total_work
        proportion_std_per_df = data.groupby(["df_id", "index"]).std()
        proportion_std = proportion_std_per_df.groupby("index").mean()[proportion_column]
    else:
        proportion_means = data.groupby("index").mean()[proportion_column]
        proportion_std = data.groupby("index").std()[proportion_column]
        average_total_work = data.groupby("index").mean().total_work

    x1 = proportion_means.index.values
    x1 = np.arange(len(x1) + 1)
    y1 = proportion_means.values
    std = proportion_std.values

    # In the beginning we can only make a random evaluation decision
    # when no samples have been evaluated
    y1 = np.insert(y1, 0, 0.5)
    std = np.insert(std, 0, 0)
    ci = c.get_ci(x1, c.DELTA)

    decision, index, decision_bound = c.get_decision(x1, y1, ci)

    c.delta_analysis_v2(env, c.MAX_THREE_WORKER_NAME, batch_name, data, proportion_column, color)
    # if decision can be made then we retrieve the average number of samples needed
    # and number of labels to reach conclusion
    num_samples = 0
    avg_labels = 0
    better_model = ""
    if decision >= 0:
        num_samples = decision
        avg_labels = int(average_total_work.iloc[index])
        # plot the threshold line
        ax.axvline(x=decision, ymin=0, ymax=1,
                   label=f'samples={num_samples}, avg labels={avg_labels}',
                   linestyle='--', color=color)
        better_model = c.get_better_model(y_axis, decision_bound)
        ax.legend()
    else:
        num_samples = x1[-1]
        avg_labels = int(average_total_work.iloc[-1])

    ax.fill_between(x1, np.maximum(np.zeros(len(x1)), (y1 - ci)),
                    np.minimum(np.ones(len(x1)), (y1 + ci)), color=color, alpha=.1)
    # ax.fill_between(x1, (y1 - std), np.minimum(np.ones(len(x1)), (y1 + std)), color=color, alpha=c.ALPHA_STD)
    ax.plot(x1, y1, color=color)
    plt.xlim([0, max(x1)])
    plt.ylim([0, 1])

    title = 'Max 3 Workers'
    plt.title(title)
    name = f"{model_key}_max_three_worker_human_evaluation_with_ci_{c.PROBABILITY}_{c.ITERATIONS}_iter"
    c.save_plot(env, fig, batch_name, name)
    c.save_result(env, result_file, decision, model_mapper, better_model, num_samples, avg_labels, c.MAX_THREE_WORKER_NAME)


def plot_max_three_labelling_effort(batch_name, y_axis, model_mapper, color="C0", is_rankme=False):
    model_key = "model1"
    if y_axis == "model2":
        model_key = "model2"

    proportion_column = model_mapper[model_key].model_column

    data = pd.read_csv(pathlib.Path("dataframes", batch_name, c.MAX_THREE_WORKER_FILE_NAME))
    iterations = data.iteration.unique()

    num_label_effort_per_decision = []
    num_req_per_decision = []
    results = []
    for i in iterations:
        df_iteration = data[data['iteration'] == i]
        proportions = df_iteration[proportion_column].tolist()
        n_req = len(proportions)
        n = np.arange(0, n_req+1)
        proportions = np.insert(proportions, 0, 0.5)
        ci = c.get_ci(n, c.DELTA)
        decision, index, decision_bound = c.get_decision(n, proportions, ci)
        better_model = ""
        if decision >= 0:
            num_requests = decision
            label_effort = df_iteration.total_work.iloc[decision-1]
            better_model = c.get_better_model(y_axis, decision_bound)
        else:
            num_requests = n[-1]
            label_effort = df_iteration.total_work.iloc[-1]

        results_entry = {
            'iteration_id': i,
            'method': c.MAX_THREE_WORKER_LABEL,
            'num_requests': num_requests,
            'label_effort': label_effort,
            'delta': c.DELTA,
            'is_decision': decision >= 0,
            'model1': model_mapper['model1'].model_name,
            'model2': model_mapper['model2'].model_name,
            'better_model': better_model,
        }
        results.append(results_entry)
        if decision >= 0:
            num_label_effort_per_decision.append(label_effort)
            num_req_per_decision.append(num_requests)

    average_effort = np.mean(num_label_effort_per_decision)
    std_effort = np.std(num_label_effort_per_decision)

    average_req = np.mean(num_req_per_decision)
    std_req = np.std(num_req_per_decision)

    lower, upper = c.bootstrap_resample_ci(num_label_effort_per_decision)
    # sample mean - the lowest difference
    ci_upper = average_effort - lower
    # sample mean - the highest difference
    ci_lower = average_effort - upper

    decisions = [1 for i in results if i['is_decision'] == True]

    # append average information
    label_effort_result = {
        'method_id': c.MAX_THREE_ID,
        'method': c.MAX_THREE_WORKER_LABEL,
        'average_effort': average_effort,
        'std_effort': std_effort,
        'average_req': average_req,
        'std_req': std_req,
        'ci_upper': ci_upper,
        'ci_lower': ci_lower,
        'ci': '' if math.isnan(ci_lower) else f'{round(ci_lower)}--{round(ci_upper)}',
        'avg_std': '' if math.isnan(average_effort) else f'${round(average_effort)} \pm {round(std_effort)}$',
        'delta': c.DELTA,
        'decisions': len(decisions)
    }

    c.update_labelling_effort(batch_name, label_effort_result)

    # write the results for each iteration run
    folder = pathlib.Path(c.CSV_RESULTS_FOLDER, batch_name, str(c.DELTA), 'label_effort_results')
    if not os.path.exists(pathlib.Path(folder)):
        os.makedirs(pathlib.Path(folder))

    df = pd.DataFrame(results)
    df.to_csv(pathlib.Path(folder, f'label_effort_{c.MAX_THREE_ID}'), index=False)
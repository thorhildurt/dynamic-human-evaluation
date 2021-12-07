import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import shared_functions as c
import os
import math


def plot_mv_with_ci(env, batch_name, y_axis, model_mapper, result_file, num_votes=5, color="C0"):
    model_key = "model1"
    if y_axis == "model2":
        model_key = "model2"
    model_id = model_mapper[model_key].model_id
    proportion_column = model_mapper[model_key].model_column

    data = pd.read_csv(pathlib.Path("dataframes", batch_name, c.MV_WORKER_FILE_NAME.format(num_votes)))
    fig, ax = plt.subplots(figsize=(4, 3))

    plt.axhline(0.5, color='gray')
    ax.set_ylabel(f'Mean Proportion for Model {model_mapper[model_key].label_name}')
    ax.set_xlabel('Number of Request Pairs')

    proportion_means = data.groupby("index").mean()[proportion_column]
    proportion_std = data.groupby("index").std()[proportion_column]
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

    c.delta_analysis_v2(env, c.MV_WORKER_NAME.format(num_votes), batch_name, data, proportion_column, color)
    # if decision can be made then we retrieve the average number of samples needed
    # and number of labels to reach conclusion
    num_samples = 0
    avg_labels = 0
    better_model = ""
    if decision >= 0:
        num_samples = decision
        avg_labels = num_votes * decision
        # plot the threshold line
        ax.axvline(x=decision, ymin=0, ymax=1,
                   label=f'samples={decision}, avg labels={num_votes * decision}',
                   linestyle='--', color=color)
        better_model = c.get_better_model(y_axis, decision_bound)
        ax.legend()
    else:
        num_samples = x1[-1]
        avg_labels = x1[-1] * num_votes

    ax.fill_between(x1, np.maximum(np.zeros(len(x1)), (y1 - ci)),
                    np.minimum(np.ones(len(x1)), (y1 + ci)), color=color, alpha=.1)
    # ax.fill_between(x1, (y1 - std), np.minimum(np.ones(len(x1)), (y1 + std)), color=color, alpha=c.ALPHA_STD)
    ax.plot(x1, y1, color=color)
    plt.xlim([0, max(x1)])
    plt.ylim([0, 1])

    title = f'{num_votes} Workers'
    plt.title(title)
    name = f"{model_key}_mv_{num_votes}_worker_human_evaluation_with_ci_{c.PROBABILITY}_{c.ITERATIONS}_iter"
    c.save_plot(env, fig, batch_name, name)
    c.save_result(env, result_file, decision, model_mapper, better_model, num_samples, avg_labels,
                  c.MV_WORKER_NAME.format(num_votes))


def plot_mv_labelling_effort(batch_name, y_axis, model_mapper, num_votes=5, color="C0"):
    model_key = "model1"
    if y_axis == "model2":
        model_key = "model2"

    proportion_column = model_mapper[model_key].model_column

    data = pd.read_csv(pathlib.Path("dataframes", batch_name, c.MV_WORKER_FILE_NAME.format(num_votes)))
    iterations = data.iteration.unique()

    num_label_effort_per_decision = []
    num_req_per_decision = []
    results = []
    for i in iterations:
        df_iteration = data[data['iteration'] == i]
        proportions = df_iteration[proportion_column].tolist()
        n_req = len(proportions)
        n = np.arange(0, n_req + 1)
        proportions = np.insert(proportions, 0, 0.5)
        ci = c.get_ci(n, c.DELTA)
        decision, index, decision_bound = c.get_decision(n, proportions, ci)
        better_model = ""
        if decision >= 0:
            num_requests = decision
            label_effort = num_votes * decision
            better_model = c.get_better_model(y_axis, decision_bound)
        else:
            num_requests = n[-1]
            label_effort = n[-1] * num_votes

        results_entry = {
            'iteration_id': i,
            'method': c.MV_WORKER_LABEL.format(num_votes),
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
        'method_id': c.MV_ID.format(num_votes),
        'method': c.MV_WORKER_LABEL.format(num_votes),
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
    df.to_csv(pathlib.Path(folder, f'label_effort_{c.MV_ID.format(num_votes)}'), index=False)

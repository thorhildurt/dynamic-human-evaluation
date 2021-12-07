import numpy as np
import evaluation_methods as em
import plot_functions as pf
import dataframe_helpers as dfh
from tqdm import tqdm
import os
import argparse
import time
import pandas as pd
import pathlib
from dask import dataframe as dd

ONE_WORKER = True
ONE_R_WORKER = True
N_W_MV = True
MAX_3 = True

PLOT_COLORS = {
    "MV_5": "C0",
    "MV_7": "C1",
    "MAX3": "C2",
    "ONE": "C3",
    "ONE_R": "C4"
}

REQ_SAMPLE_SIZE = 30000


def get_method_ids(n_workers_mv):
    methods = []
    if N_W_MV:
        for n in n_workers_mv:
            methods.append(f"MV_{n}")
    if MAX_3:
        methods.append(f"MAX3")
    if ONE_WORKER:
        methods.append(f"ONE")
    if ONE_R_WORKER:
        methods.append(f"ONE_R")
    return methods


def get_exp_id():
    folder = 'dataframes'
    blocked_ids = [f.path.split('_')[1] for f in os.scandir(folder) if f.is_dir() and 'run' in str(f)]
    ids = np.arange(1000)
    for i in blocked_ids:
        num = int(i)
        idx = np.where(ids == num)[0]
        ids = np.delete(ids, idx)
    return np.random.choice(ids)


def save_requests(req_folder, requests, mean, var, std, sample_size, seed):
    rows = []
    for i in requests:
        row = {
            'request': i,
            'mean': mean,
            'std': std,
            'var': round(std ** 2, 2)
        }
        rows.append(row)

    dataframes_folder = pathlib.Path("dataframes")
    if not os.path.exists(pathlib.Path(dataframes_folder, req_folder)):
        os.makedirs(pathlib.Path(dataframes_folder, req_folder))

    df = pd.DataFrame(rows)
    df.to_csv(pathlib.Path(dataframes_folder, req_folder, f'{sample_size}_mean={mean}_var={var}_seed={seed}'),
              index=False)


def simulate_label_aggregation_methods(test_folder, seed, d_means, var, number_of_requests,
                                       project_iterations, n_workers_mv, annotation_workers, procedure):
    np.random.seed(seed)
    for idx, mean in enumerate(d_means):
        n_requests = number_of_requests[idx]
        df_requests = pd.read_csv(
            pathlib.Path('dataframes', 'requests', f'{REQ_SAMPLE_SIZE}_mean={mean}_var={var}_seed={seed}'))
        requests = df_requests.request[:n_requests].tolist()

        print(f'\nEvaluation procedure: {procedure}, Difficulty mean = {mean}')
        dfc = dfh.DataframeCollections()
        for p in tqdm(range(project_iterations)):
            if N_W_MV:
                # n workers majority vote per request over all requests
                for w in n_workers_mv:
                    em.n_workers_evaluation_mv(dfc, p, mean, annotation_workers, requests, n=w, procedure=procedure)
            if MAX_3:
                # tvo labels per request. If there is a conflict we dynamically add 1 worker
                em.conflict_resolution(dfc, p, mean, annotation_workers, requests, procedure=procedure)
            if ONE_WORKER:
                # the same random worker evaluates all requests
                em.one_worker_evaluation(dfc, p, mean, annotation_workers, requests, procedure=procedure)
            if ONE_R_WORKER:
                # one random worker evaluates each request
                em.one_random_worker_evaluation(dfc, p, mean, annotation_workers, requests, procedure=procedure)

        # save simulated data
        df = pd.DataFrame(dfc.df_y_proportion_x_samples)
        df.to_csv(pathlib.Path('dataframes', test_folder, f'{procedure}_{mean}'), index=False)


def save_experiment_details(args, test_id, t, test_folder, request_seed, worker_seed):
    exp_file = "experiments.csv"
    save_time = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    columns = ['test_id', 'result_folder', 'timestamp', 'request_seed', 'worker_seed']
    values = [test_id, test_folder, save_time, request_seed, worker_seed]
    argv = vars(args)
    for i in argv.keys():
        columns.append(i)
        values.append(argv[i])
    exp_param_config = pd.DataFrame([values], columns=columns)

    if not os.path.isfile(exp_file):
        exp_param_config.to_csv(exp_file, index=False)
    else:
        df = pd.read_csv(exp_file)
        df = pd.concat([df, exp_param_config], sort=False)
        df = df.set_index('timestamp')
        df.to_csv(exp_file)


def plot_simulated_evaluation(test_id, test_folder, dataframes, d_means, var, method_ids, delta, n_iterations,
                              n_requests):
    # update to run specific plots
    proportion_per_request = False
    proportion_per_labels = True
    all_methods_with_ci = True
    visualise_avg_effort = False
    paper_plot = True

    print("Plotting results...")
    for key in dataframes:
        for i, rd_mean in tqdm(enumerate(d_means)):
            mean_test_folder = pathlib.Path(test_folder, f'mean={rd_mean}')
            n_request = n_requests[i]
            if all_methods_with_ci:
                for j, method_id in enumerate(method_ids):
                    color = PLOT_COLORS[method_id]
                    # Analyse how the proportion converges over increasing samples for each method
                    pf.plot_proportion_per_request_for_a_single_method_with_ci(test_id, mean_test_folder,
                                                                               dataframes[key],
                                                                               rd_mean, var, method_id, color, delta,
                                                                               key, n_request, n_iterations)

            if proportion_per_request:
                pf.plot_proportion_per_request_for_all_methods(test_id, mean_test_folder,
                                                               dataframes[key], rd_mean,
                                                               var, delta, key, n_request, n_iterations)
            if proportion_per_labels:
                # additionally records labelling effort for each iteration for the raw csv results
                pf.plot_proportion_per_label_effort_for_all_methods(test_id, mean_test_folder, dataframes[key], rd_mean,
                                                                    delta,
                                                                    key, n_request, n_iterations)
            if visualise_avg_effort:
                pf.plot_average_label_effort_for_decision(test_id, mean_test_folder,
                                                          dataframes[key], rd_mean,
                                                          delta, key, n_iterations)
            if paper_plot:
                # specific comparison plot for the paper
                method_id = "ONE_R"
                pf.plot_difficulty_comparison(test_id, test_folder, dataframes[key], method_id, delta, key, n_request)


# run evaluation procedures for all sampled requests and sampled workers capabilities
def run_simulation(args):
    test_id = get_exp_id()

    seed = args.seed
    np.random.seed(seed)

    n_iterations = args.n_iterations

    n_workers = args.n_workers
    min_worker_c = args.min_worker_capability
    max_worker_c = args.max_worker_capability

    # sample the capability of workers from a uniform distribution
    workers = np.random.uniform(min_worker_c, max_worker_c, n_workers)

    # number of workers allocated per vote for majority voting
    n_workers_mv = [5, 7]

    # set default delta for the error bounds
    delta = args.delta

    # initialize list of mean values an a a variance for the difficulty distribution
    var = args.var_request_difficulty
    std = np.sqrt(var)
    difficulty_means = args.mean_request_difficulties
    # list of number of requests for each request difficulty mean
    n_requests = args.n_requests

    method_ids = get_method_ids(n_workers_mv)

    t = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    if not args.create_df and (not args.simulation_id or not args.simulation_date):
        print("Missing arguments: simulation_id or simulation_date")
        return
    elif args.simulation_id and args.simulation_date:
        test_id = args.simulation_id
        t = args.simulation_date

    test_folder = f"run_{test_id}_{t}"
    dataframes_folder = pathlib.Path("dataframes")
    if not os.path.exists(dataframes_folder):
        os.makedirs(dataframes_folder)

    # create the corresponding test folder for dataframes
    if not os.path.exists(pathlib.Path(dataframes_folder, test_folder)):
        os.makedirs(pathlib.Path(dataframes_folder, test_folder))

    # run the two-choice simulation for several evaluation methods
    if args.create_df:
        # sample large enough sample initially, such that experiments can be
        # run on identical subsets of the original sample
        for mean in difficulty_means:
            path = pathlib.Path('dataframes', 'requests', f'{REQ_SAMPLE_SIZE}_mean={mean}_var={var}_seed={seed}')
            if not os.path.exists(path):
                requests = [max(-1, min(1, x)) for x in (np.random.normal(mean, std, REQ_SAMPLE_SIZE))]
                save_requests('requests', requests, mean, var, std, REQ_SAMPLE_SIZE, seed)

        print(f"Running simulation with test_id: {test_id}...")
        # perform multiple simulation runs for each distribution for all methods
        simulate_label_aggregation_methods(test_folder=test_folder,
                                           seed=seed,
                                           d_means=difficulty_means,
                                           var=var,
                                           number_of_requests=n_requests,
                                           project_iterations=n_iterations,
                                           n_workers_mv=n_workers_mv,
                                           annotation_workers=workers,
                                           procedure="two_choice")

    if args.write_plots:
        # read dataframes before visualisation
        df = dd.read_csv(pathlib.Path(dataframes_folder, test_folder, f"two_choice_{difficulty_means[0]}"))
        for i in range(1, len(difficulty_means)):
            temp_df = dd.read_csv(pathlib.Path(dataframes_folder, test_folder, f"two_choice_{difficulty_means[i]}"))
            df = df.append(temp_df)

        # add the chosen data collection method to a dictionary,
        # in case one wants to experiments with other simualted methods
        dataframes = {"two_choice": df}

        plot_simulated_evaluation(test_id, test_folder, dataframes, difficulty_means, var, method_ids, delta,
                                  n_iterations, n_requests)

    # save arguments for the corresponding experiment
    save_experiment_details(args, test_id, t, test_folder, request_seed=seed, worker_seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--n_iterations', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=100)
    parser.add_argument('--min_worker_capability', type=float, default=0.8)
    parser.add_argument('--max_worker_capability', type=float, default=1.0)
    parser.add_argument('--mean_request_difficulties', nargs="*", type=float, default=[0.25, 0.125, 0.0625])
    parser.add_argument('--n_requests', nargs="*", type=int, default=[3500, 5000, 15000])
    parser.add_argument('--var_request_difficulty', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--create_df', type=int, choices=[0, 1], default=1)
    parser.add_argument('--write_plots', type=int, choices=[0, 1], default=1)
    parser.add_argument('--simulation_id', type=str, default="")
    parser.add_argument('--simulation_date', type=str, default="")

    arguments = parser.parse_args()

    run_simulation(arguments)

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_FOLDER = "visual_analysis"

if __name__ == "__main__":
    folder = 'extra_plots'
    var = 0.1
    hard_distr_path = pathlib.Path('dataframes', 'requests', f'10000_mean=0.0625_var={var}_seed=21')
    medium_distr_path = pathlib.Path('dataframes', 'requests', f'10000_mean=0.125_var={var}_seed=21')
    easy_distr_path = pathlib.Path('dataframes', 'requests', f'20000_mean=0.25_var={var}_seed=21')

    paths = [hard_distr_path, medium_distr_path, easy_distr_path]
    plot_colors = ['red', 'orange', 'green']
    labels = [r'$\mu=0.0625$', r'$\mu=0.125$', r'$\mu=0.25$']

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axvline(x=0, ymin=0, ymax=1, color='black', linewidth=2)
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('Request Difficulties')
    for indx, p in enumerate(paths):
        distr = pd.read_csv(p)
        requests = distr['request']
        t_std = np.std(requests)
        t_mean = np.mean(requests)
        count, bins, _ = plt.hist(requests, 40, density=True, color=plot_colors[indx], alpha=0)
        gaussian = 1 / (t_std * np.sqrt(2 * np.pi)) * np.exp(-(bins - t_mean) ** 2 / (2 * t_std ** 2))
        plt.plot(bins, gaussian, linewidth=2, color=plot_colors[indx], label=labels[indx])
        ax.axvline(x=t_mean, ymin=0, ymax=max(gaussian), linestyle='--', color=plot_colors[indx], linewidth=1.5)
    ax.legend()

    name = f'request_difficulties_var={var}'
    if not os.path.exists(pathlib.Path(PLOT_FOLDER, folder)):
        os.makedirs(pathlib.Path(PLOT_FOLDER, folder))
    img_path = pathlib.Path(PLOT_FOLDER, folder, f'{name}.pdf')
    fig.tight_layout()
    fig.savefig(img_path, dpi=100, format='pdf')
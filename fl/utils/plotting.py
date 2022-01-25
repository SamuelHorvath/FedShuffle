import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import get_best_runs
import numpy as np
import os

sns.set(
    style="whitegrid", font_scale=1.2, context="talk",
    palette=sns.color_palette("bright"), color_codes=False)
# consider using TrueType fonts if submitting to a conference
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.figsize'] = (8, 6)

PLOT_PATH = '../plots/'
MARKERS = ['o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D']


def plot(exps, kind, log_scale=True, legend=None, file=None,
         x_label='communication rounds', y_label=None, last=True,
         zoom=False, bounds=None, position=None):
    fig, ax = plt.subplots()

    if zoom:
        # inset axes....
        axins = ax.inset_axes(position)
        # sub region of the original image
        x1, x2, y1, y2 = bounds
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        # axins.set_xticklabels()
        # axins.set_yticklabels()
    else:
        axins = None

    for i, exp in enumerate(exps):
        runs = get_best_runs(exp, last=last)
        plot_mean_std(ax, axins, runs, kind, i)

    if log_scale:
        ax.set_yscale('log')
    if legend is not None:
        ax.legend(legend)

    ax.set_xlabel(x_label)
    if y_label is None:
        ax.set_ylabel(kind)
    else:
        ax.set_ylabel(y_label)

    fig.tight_layout()
    if file is not None:
        os.makedirs(PLOT_PATH, exist_ok=True)
        plt.savefig(PLOT_PATH + file + '.pdf')

    if zoom:
        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.show()


def plot_mean_std(ax, axins, runs, metric_name, i):
    max_len = np.max([len(run[metric_name]) for run in runs])
    runs = [run for run in runs if len(run[metric_name]) == max_len]
    quant = np.array([run[metric_name] for run in runs])
    axis = 0

    mean = np.nanmean(quant, axis=axis)
    std = np.nanstd(quant, axis=axis)

    print(f'Output value: {mean[-1]} +- {std[-1]}')

    # x = np.arange(1, len(mean) + 1)
    preffix = metric_name.split('_')[0]
    x = np.array(runs[0][preffix + '_round'])
    ax.plot(x, mean, marker=MARKERS[i], markersize=12, markevery=len(x) // 10)
    ax.fill_between(x, mean + std, mean - std, alpha=0.4)
    if axins:
        axins.plot(
            x, mean, marker=MARKERS[i], markersize=12, markevery=len(x) // 10)
        axins.fill_between(x, mean + std, mean - std, alpha=0.4)

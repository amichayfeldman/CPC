import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import pandas as pd
import re


def get_correlation_matrix(data_a, name_a, data_b, name_b, title_str):
    corr_mat = np.corrcoef(x=data_a, y=data_b)
    ax = sns.heatmap(corr_mat, annot=True, fmt="f", linewidths=0.3)
    ax.set_title(title_str)
    plt.show()


def get_dataset_statistic(csv_list):
    fig, axs = plt.subplots(len(csv_list), 1)
    for i, csv_path in enumerate(csv_list):
        df = pd.read_csv(csv_path)
        df.columns = ["a", "setup_id", "name"]
        df.drop("a", axis=1)
        #         names = df.name.unique()
        #         df.groupby("name").count()
        ax = axs[i]
        counts = df.name.value_counts()
        names = counts.keys().to_list()
        names = [s.replace(' ', '\n') for s in names]
        # df.name.value_counts().hist(ax=ax)
        axs[i].bar(names, counts, align='center', width=0.2)
        axs[i].set_title('{}'.format(os.path.basename(csv_path)))
        axs[i].tick_params(labelsize=5)
        # pd.DataFrame.hist(data=df["name"], ax=ax)
        # df.plot.hist(column="name", ax=ax, title=os.path.basename(csv_path))

    plt.tight_layout()
    # plt.tick_params(labelsize=4)
    plt.show()
    x=1


def show_dataset_metric(dataset, metrics):
    """
    :param dataset: with shape (#seq, f_axis, num_of_examples)
    """
    if type(metrics) is list:
        axs = plt.subplots(len(metrics), 1)
    else:
        axs = plt.subplots(1, 1)

    i = 0
    if 'max_f' in metrics:
        max_f = np.max(np.argmax(dataset > 0, axis=1), axis=0).view(1, -1)
        axs[i].hist(max_f, bins=10, title='max f')
        i += 1
    if 'min_f' in metrics:
        min_f = np.min(np.argmin(dataset > 0, axis=1), axis=0).view(1, -1)
        axs[i].hist(min_f, bins=10, title='min f')
        i += 1
    if 'mean_f' in metrics:
        mean_f = np.mean(np.mean(dataset, axis=1), axis=0).view(1, -1)
        axs[i].hist(mean_f, bins=10, title='mean f')
        i += 1
        std_f = np.std(np.mean(dataset, axis=1), axis=0).view(1, -1)
        axs[i].hist(std_f, bins=10, title='std f')
        i += 1
    if 'mean_argmax_f':
        mean_argmax_f = np.mean(np.argmax(dataset > 0, axis=1), axis=0).view(1, -1)
        axs[i].hist(mean_argmax_f, bins=10, title='mean of max f')
        i += 1
        std_argmax_f = np.std(np.argmax(dataset > 0, axis=1), axis=0).view(1, -1)

        b = axs[i].bar(x=np.arange(dataset.shape[2]), height=mean_argmax_f)
        for i, rect in enumerate(b):
            height = b.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{}'.format(round(std_argmax_f[0, i], 2)),
                     ha='center', va='bottom')
        i += 1
    if 'mean_bw':
        bw = np.mean(np.argmax(dataset > 0, axis=1) - np.argmin(dataset > 0, axis=1), axis=0).view(1, -1)
        bw_std = np.std(np.argmax(dataset > 0, axis=1) - np.argmin(dataset > 0, axis=1), axis=0).view(1, -1)

        b = axs[i].bar(x=np.arange(dataset.shape[2]), height=bw)
        for i, rect in enumerate(b):
            height = b.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{}'.format(round(bw_std[0, i], 2)), ha='center',
                     va='bottom')
        i += 1


def main():
    main_data_folder = '/Neteera/Work/amichay.feldman/datasets/rest'
    csv_list = glob.glob(os.path.join(main_data_folder, '*.csv'))

    get_dataset_statistic(csv_list=csv_list)

    for csv_path in csv_list:
        dataset_name = os.path.basename(csv_path)
        name = dataset_name[dataset_name.find('_')+1:dataset_name.rfind('_')]
        npy_files_list = glob.glob(os.path.join(main_data_folder, '{}_data'.format(name), '*.npy'))
        df = pd.read_csv(csv_path)
        data = []
        for npy_path in npy_files_list:
            data.append(np.load(npy_path).reshape(-1))

        # get_correlation_matrix(data_a=data, name_a='train', data_b=data, name_b='train', title_str='train corr')

if __name__ == '__main__':
    main()

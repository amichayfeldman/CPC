
import csv
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def write_to_csv(path, data_list):
    df = pd.DataFrame(data_list).transpose()
    df.columns = ['Epoch', 'train_loss', 'val_loss', 'lr', 'wd']
    df.Epoch = df.Epoch.astype(int)
    df.to_csv(path)
    plt.figure()
    best_epoch = [np.argmin(df.val_loss.to_numpy())]
    plt.plot(df.Epoch.to_numpy(), np.log10(df.train_loss.to_numpy()), '-rD',  label='train_NCE_loss', markevery=best_epoch)
    plt.plot(df.Epoch.to_numpy(), np.log10(df.val_loss.to_numpy()), '-gD', label='val_NCE_loss',  markevery=best_epoch)

    plt.title("train and val NCE loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("NCE loss [log scale]")
    plt.savefig(os.path.join(os.path.dirname(path), 'losses.jpg'))
    # writer = csv.writer(open(path, 'w', newline=""))
    # writer.writerows(data_list)


def freeze_blocks(block):
    for child in block.children():
        if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.BatchNorm2d):
            child.weight.requires_grad = False
            if child.bias is not None:
                child.bias.requires_grad = False
            else:
                freeze_blocks(child)


def train_blocks(block):
    for child in block.children():
        if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.BatchNorm2d):
            child.weight.requires_grad = True
            if child.bias is not None:
                child.bias.requires_grad = True
            else:
                train_blocks(child)


def write_results(csv_path, results):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame(results)])
    os.remove(csv_path)
    df.to_csv(csv_path)


def plot_single_sample(sample_in, sample_out, output_path):
    seq_in_len = sample_in.shape[0]
    if len(sample_out.shape) == 1:
        sample_out = sample_out.unsqueeze(dim=0)

    seq_out_len = sample_out.shape[0]
    fig, axs = plt.subplots(max(seq_in_len, seq_out_len), 2, sharex=True)
    for i in range(seq_in_len):
        axs[i, 0].plot(np.arange(sample_in.shape[1]), sample_in[i, :])
        axs[i, 0].set_xlabel('freq[Hz]')
        axs[i, 0].set_ylabel('PDF')
        axs[i, 0].grid(True)
        axs[i, 0].set_title("seq_in_#{}".format(i))

    for i in range(seq_out_len):
        axs[i, 1].plot(np.arange(sample_out.shape[1]), sample_out[i, :])
        axs[i, 1].set_xlabel('freq[Hz]')
        axs[i, 1].set_ylabel('PDF')
        axs[i, 1].grid(True)
        axs[i, 1].set_title("seq_output_#{}".format(i))

    for idx in range(seq_in_len-seq_out_len):
        axs[i+idx + 1, 1].remove()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_dataset(train_dl, val_dl, config):
    output_folder = os.path.join(config['Paths']['output_folder'], 'plotted_datasets')
    if not os.path.isdir(os.path.join(output_folder, 'train')):
        os.makedirs(os.path.join(output_folder, 'train'))
    if not os.path.isdir(os.path.join(output_folder, 'val')):
        os.makedirs(os.path.join(output_folder, 'val'))

    for i, data in enumerate(train_dl):
        train_dl.dataset.change_seq_len()
        seq_in, seq_out, setup_ids = data['signal_in'], data['gt'], data['setup_id']
        for s_in, s_out, id in zip(seq_in, seq_out, setup_ids):
            name = id[:id.find('.')]
            out_path = os.path.join(output_folder, 'train', '{}.jpg'.format(name))
            plot_single_sample(sample_in=s_in.squeeze(), sample_out=s_out.squeeze(), output_path=out_path)

        if i * config['Params']['batch_size'] > 150:
            break

    for i, data in enumerate(val_dl):
        val_dl.dataset.change_seq_len()
        seq_in, seq_out, setup_ids = data['signal_in'], data['gt'], data['setup_id']
        for s_in, s_out, id in zip(seq_in, seq_out, setup_ids):
            name = id[:id.find('.')]
            out_path = os.path.join(output_folder, 'val', '{}.jpg'.format(name))
            plot_single_sample(sample_in=s_in.squeeze(), sample_out=s_out.squeeze(), output_path=out_path)

        if i * config['Params']['batch_size'] > 150:
            break







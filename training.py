from Utils.train_funcs import train_model
from Dataset.dataset import get_dataloaders
from Configuration import *
from Utils.losses import NceLoss
from Models.CDCK2 import CDCK2
from Utils.train_funcs import train_model
from Utils.help_funcs import write_results, plot_dataset

import torch
import os
import numpy as np
import glob

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(3)
np.random.seed(2)


def main():
    model_cpc = CDCK2(prediction_timestep=config['Params']['prediction_timestep'],
                      hidden_size=config['Params']['GRU_hidden_size'],
                      z_len=config['Params']['z_len'],
                      batch_size=config['Params']['batch_size'])
    train_dataloader, val_dataloader, _ = get_dataloaders(config=config)
    if config['Params']['plot_before_training']:
        plot_dataset(train_dl=train_dataloader, val_dl=val_dataloader, config=config)

    dataloaders = {'train_dl': train_dataloader, 'val_dl': val_dataloader}
    loss = NceLoss()

    nce_train, nce_val, acc_train, acc_val, lr, wd = train_model(model=model_cpc, data_loaders_dict=dataloaders,
                                                                 config=config, loss_criterion_nce=loss,
                                                                 save_model=False, write_csv=False)

    # # --- save results: --- #
    # output_folder = config['Paths']['output_folder']
    # csv_path = glob.glob(os.path.join(output_folder, '*.csv'))[0]
    # if len(csv_path) > 0:

    # data = [task_name, lr, wd, nce_train, nce_val, acc_train, acc_val]
    # write_results(csv_path=os.path.join(os.path.dirname(os.path.dirname(config['Paths']['output_folder'])), 'runnings.csv'), results=data)


if __name__ == '__main__':
    main()


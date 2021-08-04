import h5py
import numpy as np
import torch
import os
import glob
import pandas as pd


class CpcDataset(torch.utils.data.Dataset):
    def __init__(self, setups_data_path,  prediction_timestep, mode):
        self.mode = mode
        self.setups_data = pd.read_csv(setups_data_path)
        self.main_folder = os.path.dirname(setups_data_path)
        self.prediction_timestep = prediction_timestep
        self.seq_len = None

        self.names = self.setups_data['user_name'].unique()
        self.change_seq_len()

    def __len__(self):
        return len(self.setups_data)

    def change_seq_len(self):
        self.seq_len = torch.randint(low=3, high=8, size=(1,))

    def __getitem__(self, idx):
        # generate random row:
        rand_name = np.random.choice(self.names, size=1)[0]
        relevant_df = self.setups_data.loc[self.setups_data['user_name'] == rand_name]
        row = relevant_df.sample(n=1, axis=0)

        # row = self.setups_data.iloc[idx]
        data_path = os.path.join(self.main_folder, '{}_data'.format(self.mode), 'pdfs_setup_{}.npy'.format(
                                 row['setup_id'].to_string(index=False).strip()))
        data = np.load(data_path)
        start_t = torch.randint(low=0, high=data.shape[0]-self.prediction_timestep-int(self.seq_len), size=(1,))
        seq_in = data[start_t:start_t+self.seq_len, :]
        seq_out = data[start_t+self.seq_len:start_t+self.seq_len+self.prediction_timestep, :]

        # check if input is normalized:
        if np.sum(seq_in.max(axis=1) > 1) > 0:
            seq_in = torch.nn.functional.softmax(torch.from_numpy(seq_in), dim=1)
            seq_out = torch.nn.functional.softmax(torch.from_numpy(seq_out), dim=1)

        if self.mode == 'test':
            return {'signal_in': np.expand_dims(seq_in, 0), 'gt': np.expand_dims(seq_out, 0), 'idx': idx,
                    'setup_id': os.path.basename(data_path)}
        else:
            sample = {'signal_in': np.expand_dims(seq_in, 0), 'gt': np.expand_dims(seq_out, 0), 'idx': idx,
                      'setup_id': os.path.basename(data_path)}
            return sample


def get_dataloaders(config, test=False):
    data_folder = config['Paths']['data_folder']
    batch_size = config['Params']['batch_size']
    prediction_timestep = config['Params']['prediction_timestep']

    train_dataset = CpcDataset(setups_data_path=glob.glob(os.path.join(data_folder, '*train_data.csv'))[0], mode='train',
                               prediction_timestep=prediction_timestep)
    val_dataset = CpcDataset(setups_data_path=glob.glob(os.path.join(data_folder, '*val_data.csv'))[0], mode='val',
                             prediction_timestep=prediction_timestep)
    if test:
        test_dataset = CpcDataset(setups_data_path=glob.glob(os.path.join(data_folder, '*test_data.csv'))[0], mode='test',
                                  prediction_timestep=prediction_timestep)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if test:
        test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader
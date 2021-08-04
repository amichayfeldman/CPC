from Utils.help_funcs import write_to_csv

import torch
import os
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def save_model_checkpoint(model, epoch, output_path, best=False):
    """
    Saving function for training process. If the current epoch is the best, delete the previous best and save
    current model with 'Best' postfix.
    :param model: torch model instance.
    :param epoch: Int. Current epoch number is attached to the file name.
    :param output_path: Str. Where save the model.
    :param best: Bool. If current epoch is the best or not.
    """
    if best:
        previous_best_pt = glob.glob(os.path.join(output_path, '*BEST.pt'))
        if len(previous_best_pt) > 0:
            os.remove(previous_best_pt[0])
        name = os.path.join(output_path, 'model_state_dict_epoch={}_BEST.pt'.format(epoch))
    else:
        name = os.path.join(output_path, 'model_state_dict_epoch={}.pt'.format(epoch))
    torch.save(model.state_dict(), name)


def train_model(model, data_loaders_dict, config, loss_criterion_nce, save_model=True, write_csv=False):
    """
    The entire training process (train&val) for the input model is done in this function,
    include plotting results to the terminal and to a CSV. Model saving is done when we achieve the best
    performance so far.
    :param model: torch model instance.
    :param data_loaders_dict: Dict. The keys are names of the dataloaders and the values are torch Dataloader instances.
    :param config: Dict. A config instance imported from Configuration.py or from previously saved config.
    :param loss_criterion_nce: A NCE loss instance.
    :param save_model: Bool. If save models during the training or just return the best.
    :param write_csv: Bool. If save a results CSV at training final.
    """
    # # # --- Params: --- # # #
    lr = config['Params']['lr']
    wd = config['Params']['wd']
    alpha = config['Params']['alpha']
    epochs = config['Params']['epochs']
    output_folder = config['Paths']['output_folder']

    if save_model and not os.path.isdir(os.path.join(output_folder, 'saved_checkpoints')):
        os.makedirs(os.path.join(output_folder, 'saved_checkpoints'))
    elif write_csv and not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    train_dataloader, val_dataloader = data_loaders_dict['train_dl'], data_loaders_dict['val_dl']
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list, lr_list, wd_list = [], [], [], []
    best_train_loss, best_val_loss = np.inf, np.inf

    if write_csv:
        counter = 0
        csv_path = os.path.join(output_folder, "results_{}.csv")
        while os.path.isfile(csv_path.format(counter)):
            counter += 1
        csv_path = csv_path.format(counter)

    writer = SummaryWriter(comment='{}'.format(os.path.basename(config['Paths']['output_folder'])))
    ##################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    for epoch in range(epochs):
        running_loss = 0.0
        train_acc, val_acc = 0, 0
        # --- TRAIN:  --- #
        model.change_mode('train')
        for i, data in enumerate(train_dataloader):
            train_dataloader.dataset.change_seq_len()
            seq_in, seq_out = data['signal_in'].to(device), data['gt'].to(device)
            optimizer.zero_grad()
            c_t, _, z_out = model(seq_in.type(torch.FloatTensor).to(device), seq_out.type(torch.FloatTensor).to(device))
            nce_loss, batch_acc = loss_criterion_nce(c_t=c_t, z_out=z_out, w_k_list=model.Wk,
                                                     prediction_timestep=config['Params']['prediction_timestep'])
            nce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gru.parameters(), max_norm=1)
            optimizer.step()
            running_loss += nce_loss.item()
            train_acc += batch_acc
        train_loss = running_loss / (i + 1)
        train_acc /= (i + 1)
        train_loss_list.append(train_loss)
        train_acc_list.append(batch_acc)
        ##################

        model.change_mode('eval')
        # --- VAL:  --- #
        with torch.no_grad():
            val_running_loss = 0.0
            for val_i, val_data in enumerate(val_dataloader):
                val_dataloader.dataset.change_seq_len()
                seq_in, seq_out = val_data['signal_in'].to(device), val_data['gt'].to(device)
                c_t, _, z_out = model(seq_in.type(torch.FloatTensor).to(device),
                                      seq_out.type(torch.FloatTensor).to(device))
                nce_loss, batch_acc = loss_criterion_nce(c_t=c_t, z_out=z_out, w_k_list=model.Wk,
                                                         prediction_timestep=config['Params']['prediction_timestep'])
                val_running_loss += nce_loss.item()
                val_acc += batch_acc
            val_loss = val_running_loss / (val_i + 1)
            val_acc /= (val_i + 1)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
        ##################

        # --- Track results ---#
        writer.add_scalar(tag='Loss/NCE_train_loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='Loss/NCE_val_loss', scalar_value=val_loss, global_step=epoch)
        writer.add_scalar(tag='Acc/train_acc', scalar_value=train_acc, global_step=epoch)
        writer.add_scalar(tag='Acc/val_acc', scalar_value=val_acc, global_step=epoch)
        writer.add_hparams({'lr': optimizer.param_groups[-1]['lr'], 'wd': optimizer.param_groups[-1]['weight_decay']},
                           {'hparams/train_acc': train_acc, 'hparams/val_acc': val_acc,
                            'hparams/train_NCE': train_loss, 'hparams/val_NCE': val_loss})
        lr_list.append(optimizer.param_groups[-1]['lr'])
        wd_list.append(optimizer.param_groups[-1]['weight_decay'])
        ##############################

        # --- Save model checkpoint ---#
        if val_loss < best_val_loss:
            best_model = model
            best_val_loss = val_loss
            corresponding_train_loss = train_loss
            best_values = {'Epoch': epoch, 'train_nce_loss': corresponding_train_loss,
                           'val_nce_loss': best_val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
            corresponding_train_acc, corresponding_val_acc = train_acc, val_acc
            if save_model:
                save_model_checkpoint(model=model, epoch=epoch,
                                      output_path=os.path.join(output_folder, 'saved_checkpoints'), best=True)
        elif epoch % 10 == 0:
            if save_model:
                save_model_checkpoint(model=model, epoch=epoch,
                                      output_path=os.path.join(output_folder, 'saved_checkpoints'), best=False)
        ##############################

        if scheduler is not None:
            scheduler.step(val_loss)

        print("Epoch {}: train loss: {:.3f}, val loss: {:.3f}".format(epoch, train_loss, val_loss))
        print("\t\t train acc: {:.2f}, val acc : {:.2f}".format(train_acc, val_acc))

    if write_csv:
        write_to_csv(csv_path, [list(range(epochs)), train_loss_list, val_loss_list, lr_list, wd_list])

    print("\n\n\n # --- Best epoch: --- #")
    for key, value in best_values.items():
        print("{}:  {}".format(key, value))
    writer.close()
    return corresponding_train_loss, best_val_loss, corresponding_train_acc, corresponding_val_acc, best_model

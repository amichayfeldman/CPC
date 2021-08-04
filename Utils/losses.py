import torch
import numpy as np
from Configuration import *


class NceLoss(torch.nn.Module):
    def __init__(self):
        super(NceLoss, self).__init__()

        self.softmax = torch.nn.Softmax(dim=0)
        self.lsoftmax = torch.nn.LogSoftmax(dim=0)

    def forward(self, c_t, z_out, w_k_list, prediction_timestep):
        batch_size = c_t.shape[0]
        batch_nce = 0
        pred = torch.empty((prediction_timestep, batch_size, z_out.shape[-1])).float()
        for i in np.arange(0, prediction_timestep):
            linear = w_k_list[i]
            pred[i] = linear(c_t)
        for k in np.arange(0, prediction_timestep):
            total = torch.mm(z_out[:, k, ...], torch.transpose(pred[k], 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch_size)))
            if type(config['Params']['focal_loss_gamma']) == int or type(config['Params']['focal_loss_gamma']) == float:
                focal_weighting_mat = torch.eye(total.shape[0]) * torch.pow((1 - torch.diag(self.softmax(total))),
                                                                  torch.Tensor([config['Params']['focal_loss_gamma']]))
                focal_weighting_mat = torch.where(focal_weighting_mat == 0, torch.Tensor([1]), focal_weighting_mat)
                focal_res_batch = torch.mul(focal_weighting_mat, self.lsoftmax(total))
                batch_nce += torch.sum(torch.diag(focal_res_batch))
            else:
                batch_nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        batch_nce /= -1. * batch_size * prediction_timestep
        batch_acc = 1. * correct.item() / batch_size

        return batch_nce, batch_acc

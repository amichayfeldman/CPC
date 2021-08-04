import torch
import numpy as np
from Utils.help_funcs import freeze_blocks, train_blocks
from Models.help_units import conv_block
from Models.help_units import InceptionBlock


def _weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


class CDCK2(torch.nn.Module):
    def __init__(self, prediction_timestep, hidden_size, z_len, batch_size):

        super(CDCK2, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.prediction_timestep = prediction_timestep
        self.z_len = z_len
        self.hidden_size = hidden_size
        self.encoder = torch.nn.Sequential(
            InceptionBlock(in_channels=1, num_kernels_list=[8, 16, 4, 4]),
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            InceptionBlock(in_channels=32, num_kernels_list=[8, 16, 4, 4]),
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            InceptionBlock(in_channels=32, num_kernels_list=[4, 8, 2, 2]),
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            InceptionBlock(in_channels=16, num_kernels_list=[2, 4, 1, 1])
        )
        self.after_interpolation_1 = torch.nn.Sequential(
            # torch.nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            InceptionBlock(in_channels=8, num_kernels_list=[2, 4, 1, 1])
        )
        self.after_interpolation_2 = torch.nn.Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                                     bias=False)

        self.gru = torch.nn.GRU(input_size=z_len, hidden_size=hidden_size, num_layers=1, bidirectional=False,
                                batch_first=True)
        self.Wk = torch.nn.ModuleList([torch.nn.Linear(hidden_size, z_len) for i in range(prediction_timestep)])

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')
        # initialize Wk:
        for w in self.Wk:
            w.apply(_weights_init)

        self.encoder.apply(_weights_init)
        self.after_interpolation_1.apply(_weights_init)
        self.after_interpolation_2.apply(_weights_init)

    def init_hidden(self, batch_size, randomization='ones'):
        if randomization == 'normal':
            return torch.randn(size=(batch_size, 1, self.hidden_size)).to(self.device)
        elif randomization == 'zeros':
            return torch.zeros(batch_size, 1, self.hidden_size).to(self.device)
        else:
            return torch.ones(batch_size, 1, self.hidden_size).to(self.device)

    def forward(self, x_input, x_predicted):
        """

        :param x_input: A tensor with data until t=t'. Tensor should have dim
                  of (batch, seq_len, num_features)
        :param x_predicted: A tensor with data after t=t'. Tensor should have dim
                  of (batch, seq_len, num_features).
        :return: c_t - the last GRU's output. hidden - all GRU's hidden output. z_out - latent representation for
                 predicted chunk from the origin signal (the encoder's output for the second part of x)
        """
        batch = x_input.shape[0]
        z_input = self.encoder(x_input)
        z_predicted = self.encoder(x_predicted)
        z_in = torch.nn.functional.interpolate(input=z_input, size=(z_input.shape[2], int(self.z_len/2)),
                                               mode='bilinear', align_corners=True)
        z_in = self.after_interpolation_1(z_in)
        z_in = torch.nn.functional.interpolate(input=z_in, size=(z_input.shape[2], self.z_len), mode='bilinear',
                                               align_corners=True)
        z_in = self.after_interpolation_2(z_in)

        z_out = torch.nn.functional.interpolate(input=z_predicted, size=(z_predicted.shape[2], int(self.z_len / 2)),
                                                mode='bilinear', align_corners=True)
        z_out = self.after_interpolation_1(z_out)
        z_out = torch.nn.functional.interpolate(input=z_out, size=(z_predicted.shape[2], self.z_len),
                                                mode='bilinear', align_corners=True)
        z_out = self.after_interpolation_2(z_out)[:, 0, :, :]

        h0 = self.init_hidden(batch_size=x_input.shape[0], randomization='zeros')
        all_hiddens, hidden_out = self.gru(z_in.squeeze(1), h0.permute(1, 0, 2))
        c_t = hidden_out.view(batch, self.hidden_size)

        return c_t, all_hiddens, z_out

    def change_mode(self, mode):
        if mode == 'eval':
            for l in self.Wk:
                l.weight.requires_grad, l.bias.requires_grad = False, False
        # --- freeze GRU ---#
        for l in self.gru.parameters():
            l.requires_grad = False
            #############
        freeze_blocks(self.encoder)

        if mode == 'train':
            for l in self.Wk:
                l.weight.requires_grad, l.bias.requires_grad = True, True
            # --- train GRU ---#
            for l in self.gru.parameters():
                l.requires_grad = True
                #############
            train_blocks(self.encoder)


if __name__ == "__main__":

    model = CDCK2(prediction_timestep=2, batch_size=4)
    x = torch.ones(16, 1, 100)
    h = torch.zeros(5)
    model(x, h)
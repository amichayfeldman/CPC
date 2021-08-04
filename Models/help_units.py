import torch


def conv_block(in_kernels, out_kernels, kernel_size, stride=1, padding=0):
    return torch.nn.Sequential(torch.nn.Conv2d(in_kernels, out_kernels,
                                               kernel_size=(1, kernel_size), stride=(1, stride),
                                               padding=(0, padding), bias=False),
                               torch.nn.BatchNorm2d(out_kernels),
                               torch.nn.ReLU(inplace=True))


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, num_kernels_list):
        super(InceptionBlock, self).__init__()
        self.branch1X1 = torch.nn.Sequential(
            conv_block(in_kernels=in_channels, out_kernels=num_kernels_list[0], kernel_size=1))
        self.branch5X5 = torch.nn.Sequential(
            conv_block(in_kernels=in_channels, out_kernels=num_kernels_list[1], kernel_size=1, stride=1, padding=0),
            conv_block(in_kernels=num_kernels_list[1], out_kernels=num_kernels_list[1],
                       kernel_size=5, stride=1, padding=2))
        self.branch11X11 = torch.nn.Sequential(
            conv_block(in_kernels=in_channels, out_kernels=num_kernels_list[2], kernel_size=1, stride=1, padding=0),
            conv_block(in_kernels=num_kernels_list[2], out_kernels=num_kernels_list[2],
                       kernel_size=11, stride=1, padding=5))
        self.branch_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            conv_block(in_kernels=in_channels, out_kernels=num_kernels_list[3], kernel_size=1))

    def forward(self, x):
        x = [self.branch1X1(x),
             self.branch5X5(x),
             self.branch11X11(x),
             self.branch_pool(x)]
        x = torch.cat(x, 1)
        return x




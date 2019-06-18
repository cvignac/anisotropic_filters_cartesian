import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod
from layers import ImageChebychevConvolution


class ImageClassificationNetwork(nn.Module):
    def __init__(self, args):
        """ Network for image classification."""
        super().__init__()
        self.size = args.size

        if args.dataset == 'cifar10':
            input_chan = 3
        elif args.dataset == 'mnist':
            input_chan = 1
        else:
            raise ValueError("Dataset {} not implemented".format(args.dataset))
        # Define model size here
        units = [32, 32, 64]

        im_size = 32 if args.dataset == "cifar10" else 28  # image size
        im_size2 = int(im_size / 2)
        im_size3 = int(im_size2 / 2)

        self.conv1 = self.conv(input_chan, units[0], im_size)
        self.conv2 = self.conv(units[0], units[1], im_size2)
        self.conv3 = self.conv(units[1], units[2], im_size3)

        if args.dataset == 'cifar10':
            fc_size = 4096
        elif args.dataset == 'mnist':
            fc_size = 3136
        else:
            raise ValueError("Dataset not implemented")

        self.use_batch_norm = not args.no_batch_norm
        if self.use_batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(units[0])
            self.bn2 = torch.nn.BatchNorm2d(units[1])

        self.fc1 = nn.Linear(fc_size, 64)
        self.fc2 = nn.Linear(64, 10)

    def print_params(self):
        print('Parameters in the model:')
        for name, parameter in self.named_parameters():
            print(name, ':', parameter.shape)
        total_params = sum(p.numel() for p in self.parameters())
        print('Total number of parameters', total_params)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @abstractmethod
    def conv(self, input_units, output_units, im_size):
        pass


class ProductNet(ImageClassificationNetwork):
    def __init__(self, args):
        padding = False
        self.other_args = (args.size, args.isotropic, args.directed, padding,
                           args.use_L, args.use_chebychev)
        print("Running ProductNet.")
        print("Isotropic:", args.isotropic)
        super().__init__(args)
        self.print_params()

    def conv(self, input_units, output_units, im_size):
        return ImageChebychevConvolution(input_units, output_units, im_size, *self.other_args)


class StandardNet(ImageClassificationNetwork):
    def __init__(self, args):
        print("Running in normal mode.")
        # Find the correct padding size
        if args.size == 3:
            self.padding = 1
        elif args.size == 5:
            self.padding = 2
        else:
            raise ValueError('Padding size not implemented for filters of size {}x{}'.format(args.size, args.size))
        super().__init__(args)
        self.print_params()

    def conv(self, input_units, output_units, im_size):
        return nn.Conv2d(input_units, output_units, self.size, padding=self.padding)



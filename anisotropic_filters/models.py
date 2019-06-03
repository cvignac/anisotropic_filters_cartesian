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
            self.input_chan = 3
        elif args.dataset == 'mnist':
            self.input_chan = 1
        else:
            raise ValueError("Dataset {} not implemented".format(args.dataset))
        # Define model size here
        self.units = [32, 32, 64]

        self.im_size = 32 if args.dataset == "cifar10" else 28  # image size

        if args.dataset == 'cifar10':
            fc_size = 4096
        elif args.dataset == 'mnist':
            fc_size = 3136
        else:
            raise ValueError("Dataset not implemented")
        self.fc1 = nn.Linear(fc_size, 64)
        self.fc2 = nn.Linear(64, 10)

    def print_params(self):
        print('Parameters in the model:')
        for name, parameter in self.named_parameters():
            print(name, ':', parameter.shape)
        total_params = sum(p.numel() for p in self.parameters())
        print('Total number of parameters', total_params)

    @abstractmethod
    def forward(self, x):
        pass


class ProductNet(ImageClassificationNetwork):
    def __init__(self, args):
        super().__init__(args)
        other_args = (args.size, args.isotropic, args.directed)
        print("Running ProductNet.")
        print("Isotropic:", args.isotropic)

        self.zero_padding = False
        if self.zero_padding:
            im_size1 = self.im_size + 4
            im_size2 = int(im_size1 / 2) + 2
            im_size3 = int(im_size2 / 2) + 2
        else:
            im_size1 = self.im_size
            im_size2 = int(im_size1 / 2)
            im_size3 = int(im_size2 / 2)

        conv = ImageChebychevConvolution
        self.conv1 = conv(self.input_chan, self.units[0], im_size1, *other_args)
        self.conv2 = conv(self.units[0], self.units[1], im_size2, *other_args)
        self.conv3 = conv(self.units[1], self.units[2], im_size3, *other_args)
        self.bn1 = torch.nn.BatchNorm2d(self.units[0])
        self.bn2 = torch.nn.BatchNorm2d(self.units[1])
        self.bn3 = torch.nn.BatchNorm2d(self.units[2])

        self.print_params()

    def forward(self, x):
        if self.zero_padding:
            x = self.pad(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.unpad(x)
            x = F.max_pool2d(x, 2, 2)
            x = self.pad(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.unpad(x)
            x = F.max_pool2d(x, 2, 2)
            x = self.pad(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.unpad(x)
            x = x.contiguous()
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def pad(self, x):
        s0, s1, s2, s3 = x.shape
        size = self.size
        new_x = torch.zeros((s0, s1, s2 + 2 * size, s3 + 2 * size), dtype=x.dtype, device=x.device)
        new_x[:, :, size:-size, size:-size] = x
        return new_x

    def unpad(self, x):
        size = self.size
        return x[:, :, size:-size, size:-size]


class StandardNet(ImageClassificationNetwork):
    def __init__(self, args):
        super().__init__()
        print("Running in normal mode.")
        # Find the correct padding size
        if args.size == 3:
            padding = 1
        elif args.size == 5:
            padding = 2
        else:
            raise ValueError('Padding size not implemented for filters of size {}x{}'.format(args.size, args.size))

        self.conv1 = nn.Conv2d(self.input_chan, self.units[0], args.size, padding=padding)
        self.conv2 = nn.Conv2d(self.units[0], self.units[1], args.size, padding=padding)
        self.conv3 = nn.Conv2d(self.units[1], self.units[2], args.size, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


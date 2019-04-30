from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import ImageGCN
from torchvision import datasets, transforms
import numpy as np
import visualize_filters
from utils import EarlyStopping
import utils

debug = False
im_conv = False
model_name = '10-5-20-5'
save_path = './saved_models/{}.pt'.format(model_name)
results_path = './results/{}.txt'.format(model_name)


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        if im_conv:
            print("Standard convolutions are used")
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.conv2 = nn.Conv2d(10, 20, 5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
        elif debug:
            print("Debug mode is on")
            self.conv1 = ImageGCN(1, 2, 2, device)
            self.conv2 = ImageGCN(2, 2, 2, device)
            self.fc1 = nn.Linear(98, 20)
            self.fc2 = nn.Linear(20, 10)
        else:
            print("Running in normal mode.")
            self.conv1 = ImageGCN(1, 10, 5, device)
            self.conv2 = ImageGCN(10, 10, 5, device)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(490, 50)
            self.fc2 = nn.Linear(50, 10)

        print('Parameters in the model:')
        for name, parameter in self.named_parameters():
            print(name, ':', parameter.shape)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if debug:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def visualize_forward(self, x):
        x0 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x0)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x0[0], F.log_softmax(x, dim=1)[0]


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    nsamples = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        if debug and batch_idx > nsamples:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def mainMNIST():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before' +
                        ' logging training status')
    parser.add_argument('--load-model', type=bool, default=False, metavar='L',
                        help='load a trained model (default: False')
    parser.add_argument('--visualize', type=bool, default=False, metavar='L',
                        help='load a trained model and visualize' +
                        ' the first layer (default: False')
    parser.add_argument('--save-model', default=False, metavar='Z',
                        help='save the model in ./saved_models')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:3" if use_cuda else "cpu")
    print("Device used", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    path_to_datset = "../data"
    train_loader, test_loader = utils.load_mnist(path_to_datset, args, kwargs)

    model = Net(device).to(device)
    optimizer = optim.SGD(model.parameters(), args.lr, args.momentum)

    results = np.zeros(args.epochs)
    earlyS = EarlyStopping(mode='max', patience=3)

    if args.visualize:
        model.load_state_dict(torch.load(save_path))
        if im_conv:
            visualize_filters.main_image_conv(model)
        else:
            visualize_filters.main(model)
        for data, target in test_loader:
            visualize_filters.apply_to_example(model, data, target)
            break
        return

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        results[epoch - 1] = acc
        stop = earlyS.step(acc)
        if stop:
            print("Early stopping triggered at iteration", epoch)
            print("Stopping the training.")
            break
        if debug:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == 'conv1.coefs':
                        print(name, param.data)
    if args.save_model:
        torch.save(model.state_dict(), save_path)
    np.savetxt(results_path, results)


if __name__ == '__main__':
    mainMNIST()

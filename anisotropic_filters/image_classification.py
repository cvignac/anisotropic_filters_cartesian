# General
from pathlib import Path
# Calculus
import numpy as np
# Torch
import torch
import torch.optim as optim
import torch.nn.functional as F
# Project files
from visualize_filters import visualize, visualize_standard_conv
import utils as utils
import parser as parser
from models import StandardNet, ProductNet


def train(args, model, device, train_loader, optimizer, epoch):
    debug = False
    model.train()
    debug_batches = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        if debug and batch_idx > debug_batches:
            print("Debug mode triggered - stopping training")
            break
        # Core of training
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Visualization
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Parse the arguments
    args, device, model_name, paths, kwargs = parser.parserCIFAR()
    print(args)
    print('Name of the model:', model_name)
    print("In main", torch.cuda.current_device())
    # Find the dataset
    dataset = args.dataset
    if Path('/dataset/' + dataset).is_dir():
        print('Using folder /dataset')
        path_to_dataset = './'
    elif Path('/datasets2/' + dataset).is_dir():
        path_to_dataset = '/datasets2/' + dataset
    elif Path('../data').is_dir():
        print('Looking for the dataset locally')
        path_to_dataset = '../data'
    else:
        raise ValueError("No path to the dataset found")
    save_path = './saved_models/{}/{}.pt'.format(dataset, model_name)
    load_path = './saved_models/{}/{}.pt'.format(dataset, model_name)
    results_path = './results/{}/{}.txt'.format(dataset, model_name)

    if dataset == "cifar10":
        train_loader, test_loader = utils.load_cifar(path_to_dataset, args, kwargs)
    elif dataset == "mnist":
        train_loader, test_loader = utils.load_mnist(path_to_dataset, args, kwargs)
    else:
        raise ValueError("Dataset {} not implemented".format(dataset))

    # Case where there is no training
    if args.visualize:
        model = StandardNet(args) if args.standard else ProductNet(args)
        model.load_state_dict(torch.load(load_path))
        print('Model loaded')
        model = model.to('cpu')
        (visualize_standard_conv if args.standard else visualize)(model, model_name)
        return

    n_expe = args.experiments
    results = np.zeros((n_expe, args.epochs))
    for expe in range(n_expe):
        model = StandardNet(args) if args.standard else ProductNet(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
        if args.early_stopping:
            earlyS = utils.EarlyStopping(mode='max', patience=args.early_stopping)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            results[expe, epoch - 1] = acc
            stop = False if not args.early_stopping else earlyS.step(acc)
            if stop:
                print("Early stopping triggered at iteration", epoch)
                print("Stopping the training.")
                for i in range(epoch + 1, args.epochs + 1):
                    results[expe, i - 1] = acc
                break
        print('Experiment', expe, 'finished.')

    if args.save_results or args.save_model:
        utils.save_arguments(args.dataset, model_name)
    if args.save_model:
        torch.save(model.state_dict(), save_path)
    if args.save_results:
        np.savetxt(results_path, results)
    average_acc = np.mean(results[:, -1])
    print('All experiments done. Average accuracy:', average_acc)
    return {'loss': -average_acc}


if __name__ == '__main__':
    main()

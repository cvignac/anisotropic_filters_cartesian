import argparse
import torch
import os
import numpy as np


def build_model_name(epochs, isotropic, size, id):
    file_seed = id if id is not None else np.random.randint(int(10 ** 5))
    name = str(epochs) + 'epochs-'
    if isotropic:
        name = name + 'iso-'
    name = name + str(size) + '-' + str(file_seed)
    return name


def common_parser(parser):
    parser.add_argument('--size', type=int, help='size of the filters', default=2)

    parser.add_argument('--isotropic', action='store_true', default=False,
                        help='Constrain the filters to be isotropic')
    parser.add_argument('--experiments', type=int, default=1,
                        help='Number of experiments to average results on')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--early-stopping', type=int, default=5,
                        help='Patience -  0 to disable early stopping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    # GPU
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--gpu', type=int, help='Id of gpu device')

    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    # Print
    parser.add_argument('--log-interval', type=int, default=50,
                        help='How many batches to wait before logging training status')

    # Save
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save the trained model')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='Save the results as a txt file')
    parser.add_argument('--id', type=int, help='identifier of the experiment')


def parserCIFAR():
    parser = argparse.ArgumentParser(description="Image classification task")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='"cifar10" or "mnist"')
    parser.add_argument('--directed', action='store_true', default=False,
                        help='Use directed paths')
    parser.add_argument('--standard', action='store_true', default=False,
                        help='Train a standard CNN')
    common_parser(parser)
    args = parser.parse_args()
    return common_processing(args)


def common_processing(args):
    torch.manual_seed(args.seed)

    # Handle device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:" + str(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        device = "cpu"
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print('Device used:', device)


    model_name = build_model_name(args.epochs, args.isotropic,
                                  args.size, args.id)

    assert not (args.no_cuda and args.gpu)
    assert not (args.save_model and (args.n_experiments > 1))

    dataset = args.dataset
    paths = {'save': './saved_models/{}/{}.pt'.format(dataset, model_name),
             'load': './saved_models/{}/{}-cpu.pt'.format(dataset, model_name),
             'results': './results/{}/{}.txt'.format(dataset, model_name),
             'train_results': './results/{}/train{}.txt'.format(dataset, model_name)}
    return args, device, model_name, paths, kwargs

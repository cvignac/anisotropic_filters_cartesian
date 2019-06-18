import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

folder_path = './results/cifar10/'

def create_table():
    ''' Save a numpy array with the following structure:
                    CIFAR 10        MNIST
                mean    conf    mean    conf
            iso2
            iso8
            aniso2
            conv3
            conv5
    '''
    results = np.zeros((5, 4))
    f = 2.262 / 3   # Coefficient from the Student law
    # CIFAR 10
    # cifar_path = './results/cifar10/'
    # iso8 = np.loadtxt(cifar_path + 'graph-100epochs-iso-2-10.txt')
    # iso2 = np.loadtxt(cifar_path + 'graph-100epochs-iso-8-11.txt')
    # aniso2 = np.loadtxt(cifar_path + 'graph-100epochs-2-12.txt')
    # conv3 = np.loadtxt(cifar_path + 'standard-100epochs-3-13.txt')
    # conv5 = np.loadtxt(cifar_path + 'standard-100epochs-5-14.txt')
    # res_cifar = [iso2, iso8, aniso2, conv3, conv5]
    # for i, res in enumerate(res_cifar):
    #     results[i, 0] = np.mean(res[:, -1], axis=0)
    #     results[i, 1] = f * np.std(res[:, -1], axis=0)
    # MNIST
    mnist_path = './results/mnist/'
    # iso8 = np.loadtxt(mnist_path + 'graph-100epochs-iso-2-20.txt')
    # iso2 = np.loadtxt(mnist_path + 'graph-100epochs-iso-8-21.txt')
    aniso2 = np.loadtxt(mnist_path + '50epochs-2-132.txt')
    directed1 = np.loadtxt(mnist_path + '50epochs-1-141.txt')
    directed2 = np.loadtxt(mnist_path + '50epochs-2-142.txt')
    conv3 = np.loadtxt(mnist_path + '50epochs-3-113.txt')
    conv5 = np.loadtxt(mnist_path + '50epochs-5-115.txt')
    # res_mnist = [iso2, iso8, aniso2, conv3, conv5]
    res_mnist = [aniso2, directed1, directed2, conv3, conv5]
    for i, res in enumerate(res_mnist):
        results[i, 2] = np.mean(res[:, -1], axis=0)
        results[i, 3] = f * np.std(res[:, -1], axis=0)
    # Process results
    np.savetxt('results_images.txt', results, fmt='%.3f')
    print(results)


    # # Movielens100k
    # iso = np.array([0.939023, 0.941270, 0.938805, 0.941265, 0.936824, 0.942998, 0.938459, 0.937278, 0.935927, 0.940648])
    #
    #                 #0.944219, 0.956575, 0.954904, 0.956351, 0.955504, 0.956421, 0.952571, 0.956672, 0.954547, 0.950492 ])
    # aniso = np.array([0.932687, 0.93679, 0.93750, 0.931799, 0.932642, 0.938435, 0.931335,  0.930390, 0.937903, 0.935545])
    # print("Mean, confidence iso", np.mean(iso), f * np.std(iso))
    # print("Mean, confidence aniso", np.mean(aniso), f * np.std(aniso))
    #
    # iso_perso = np.loadtxt('./results/movielens100k/1500epochs-iso-2-13.txt')
    # aniso_perso = np.loadtxt('./results/movielens100k/1500epochs-2-12.txt')
    # movielens_results = [[], []]
    # for res in [iso_perso, aniso_perso]:
    #     movielens_results[0].append(np.mean(res[:, -1], axis=0))
    #     movielens_results[1].append(f * np.std(res[:, -1], axis=0))
    # np.savetxt('results_movielens.txt', movielens_results, fmt='%.5f')
    # print("Results on Movielens")
    # print(movielens_results)


def analyse_acc():

    iso8 = np.loadtxt(folder_path + 'graph-100epochs-iso-2-10.txt')
    iso2 = np.loadtxt(folder_path + 'graph-100epochs-iso-8-11.txt')
    aniso2 = np.loadtxt(folder_path + 'graph-100epochs-2-12.txt')
    conv3 = np.loadtxt(folder_path + 'standard-100epochs-3-13.txt')
    conv5 = np.loadtxt(folder_path + 'standard-100epochs-5-14.txt')
    res = [iso2, iso8, aniso2, conv3, conv5]

    # Process results
    mean = [np.mean(x, axis=0) for x in res]
    f = 2.262 / 3   # Coefficient from the Student law
    confidence = [f * np.std(x, 0) for x in res]

    epochs = 100
    lin = np.arange(0, epochs)
    zipped = [(x - y, x + y) for x, y in zip(mean, confidence)]
    plt.plot(lin, mean[3], c='y', label='Standard convolution, kernels 3x3')
    plt.plot(lin, mean[4], c='b', label='Standard convolution, kernels 5x5')
    plt.plot(lin, mean[2], c='r', label='Anisotropic, degree 2')
    plt.plot(lin, mean[0], c='g', label='Isotropic degree 2')
    plt.plot(lin, mean[1], c='m', label='Isotropic degree 8')
    plt.fill_between(lin, *zipped[3], alpha=0.3, color='y')
    plt.fill_between(lin, *zipped[4], alpha=0.3, color='b')
    plt.fill_between(lin, *zipped[2], alpha=0.3, color='r')
    plt.fill_between(lin, *zipped[0], alpha=0.3, color='g')
    plt.fill_between(lin, *zipped[1], alpha=0.3, color='m')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy on CIFAR10, 10 experiments')
    # plt.savefig('testaccuracyCIFAR30epochs')

    plt.show()


def compare_flip():

    f = 2.776 / 2   # Coefficient from the student law
    n_epochs = 30
    path = folder_path + 'custom-50epochs-2-'
    noyes = np.loadtxt(path + '6.txt')[:, :n_epochs]
    mean_noyes, std_noyes = np.mean(noyes, 0), f * np.std(noyes, 0)

    yesyes = np.loadtxt(folder_path + 'product-30epochs-2-126.txt')
    # yesyes = np.loadtxt(path + '7.txt')[:, :n_epochs]
    mean_yesyes, std_yesyes = np.mean(yesyes, 0), f * np.std(yesyes, 0)

    yesno = np.loadtxt(path + '8.txt')[:, :n_epochs]
    mean_yesno, std_yesno = np.mean(yesno, 0), f * np.std(yesno, 0)

    nono = np.loadtxt(path + '9.txt')[:, :n_epochs]
    mean_nono, std_nono = np.mean(nono, 0), f * np.std(nono, 0)

    two_chan = np.loadtxt(folder_path + 'directed-30epochs-2-126.txt')
    m2chan, std_2chan = np.mean(two_chan, 0), f * np.std(two_chan, 0)

    x = np.linspace(1, 30, 30)
    plt.plot(x, mean_nono, c='y', label='No flipping')
    plt.fill_between(x, mean_nono - std_nono,
                     mean_nono + std_nono, alpha=0.2, color='y')
    plt.plot(x, mean_noyes, c='b', label='Flipping at test time')
    plt.fill_between(x, mean_noyes - std_noyes,
                     mean_noyes + std_noyes, alpha=0.2, color='b')
    # plt.plot(x, mean_yesno, c='r', label='Flipping at training time')
    # plt.fill_between(x, mean_yesno - std_yesno,
    #                mean_yesno + std_yesno, alpha=0.2, color='r')
    plt.plot(x, mean_yesyes, c='m', label='Data augmentation')
    plt.fill_between(x, mean_yesyes - std_yesyes,
                     mean_yesyes + std_yesyes, alpha=0.2, color='m')
    plt.plot(x, m2chan, c='g', label='Two channels')
    plt.fill_between(x, m2chan - std_2chan, m2chan + std_2chan, alpha=0.15,
                     color='g')
    plt.fill_between
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Test Accuracy on CIFAR10 of 5 experiments')
    plt.savefig('testAccuracyFlip')
    plt.show()


# compare_flip()
if __name__ == '__main__':
    create_table()

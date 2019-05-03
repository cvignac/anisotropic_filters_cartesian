# Anisotropic filters for product graphs

Pytorch implementation of the paper:

*Learning anisotropic filters on product graphs*, Clément Vignac & Pascal Frossard, 2019


## Image classification

To display the different options:

```bash
python3 image_classification.py -h
```

The code takes too long to be run without GPU, and typically a couple of hours per experiment on a GPU. This is due to
fact that sparse semantics are not used: it appeared faster to use full matrix multiplications on a GPU rather than
sparse semantics on CPU.

Example: Isotropic filters on MNIST:

``` python3 image_classification.py --dataset mnist --isotropic --id 1 --save-results --size 2 ```



## Recommender systems

As stated in the paper, the settings and code of https://github.com/fmonti/mgcnn are used.

To create an isotropic version of their filters, replace two lines of the ```Train_test_matrix_completion``` class in https://github.com/fmonti/mgcnn/blob/master/Notebooks/movielens/supervised_approach_movielens_factorization_2_different_conv.ipynb:

```
self.W_conv_H = tf.get_variable("W_conv_H", shape=[self.ord_row*initial_W.shape[1], self.n_conv_feat],
                                initializer=tf.contrib.layers.xavier_initializer())
self.b_conv_H = tf.Variable(tf.zeros([self.n_conv_feat,]))
 ```
 become
 ```
self.W_conv_H = self.W_conv_W
self.b_conv_H = self.b_conv_W
```
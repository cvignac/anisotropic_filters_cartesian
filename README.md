# Anisotropic filters for product graphs

Pytorch implementation of the paper:

*Learning anisotropic filters on product graphs*, Clément Vignac & Pascal Frossard, 2019


## Image classification

To display the different options:

```bash
python3 image_classification.py -h
```




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
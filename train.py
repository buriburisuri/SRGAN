# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image

# corrupted image
x_small = tf.image.resize_bicubic(x, (14, 14))
x_nearest = tf.image.resize_images(x_small, (28, 28), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# generator labels ( all ones )
y = tf.ones(batch_size, dtype=tf.sg_floatx)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# create generator
#
# I've used ESPCN scheme
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
#

# generator network
with tf.sg_context(name='generator', act='relu', bn=True):
    gen = (x_small
           .sg_conv(dim=32)
           .sg_conv()
           .sg_conv(dim=4, act='sigmoid', bn=False)
           .sg_periodic_shuffle(factor=2))

# add image summary
tf.sg_summary_image(gen)

#
# input image pairs
#
x_real_pair = tf.concat(3, [x_nearest, x])
x_fake_pair = tf.concat(3, [x_nearest, gen])

#
# create discriminator & recognizer
#

# create real + fake image input
xx = tf.concat(0, [x_real_pair, x_fake_pair])

with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu'):
    # discriminator part
    disc = (xx.sg_conv(dim=64)
              .sg_conv(dim=128)
              .sg_flatten()
              .sg_dense(dim=1024)
              .sg_dense(dim=1, act='linear')
              .sg_squeeze())

#
# loss and train ops
#

loss_disc = tf.reduce_mean(disc.sg_bce(target=y_disc))  # discriminator loss
loss_gen = tf.reduce_mean(disc.sg_reuse(input=x_fake_pair).sg_bce(target=y))  # generator loss

train_disc = tf.sg_optim(loss_disc, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=20, ep_size=data.train.num_batch, early_stop=False)

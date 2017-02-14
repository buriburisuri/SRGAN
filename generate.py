# -*- coding: utf-8 -*-
import sugartensor as tf
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 30   # batch size

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=batch_size)

# input images
x = data.train.image

# corrupted image
x_small = tf.image.resize_bicubic(x, (14, 14))
x_bicubic = tf.image.resize_bicubic(x_small, (28, 28)).sg_squeeze()
x_nearest = tf.image.resize_images(x_small, (28, 28), tf.image.ResizeMethod.NEAREST_NEIGHBOR).sg_squeeze()

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
           .sg_periodic_shuffle(factor=2)
           .sg_squeeze())

#
# run generator
#

fig_name = 'asset/train/sample.png'
with tf.Session() as sess:
    with tf.sg_queue_context(sess):

        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

        # run generator
        gt, low, bicubic, sr = sess.run([x.sg_squeeze(), x_nearest, x_bicubic, gen])

        # plot result
        _, ax = plt.subplots(10, 12, sharex=True, sharey=True)
        for i in range(10):
            for j in range(3):
                ax[i][j*4].imshow(low[i*3+j], 'gray')
                ax[i][j*4].set_axis_off()
                ax[i][j*4+1].imshow(bicubic[i*3+j], 'gray')
                ax[i][j*4+1].set_axis_off()
                ax[i][j*4+2].imshow(sr[i*3+j], 'gray')
                ax[i][j*4+2].set_axis_off()
                ax[i][j*4+3].imshow(gt[i*3+j], 'gray')
                ax[i][j*4+3].set_axis_off()

        plt.savefig(fig_name, dpi=600)
        tf.sg_info('Sample image saved to "%s"' % fig_name)
        plt.close()

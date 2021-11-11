from __future__ import division
import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import argparse
import os
tf.set_random_seed(19)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from glob import glob
import time
from collections import namedtuple


"""
Some codes from https://github.com/Newmu/dcgan_code
"""

import math
import pprint
import scipy.misc
import numpy as np
import copy

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size*4])
    img = img/127.5 - 1
    return img

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False,is_noise=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    img_C = imread(image_path[2])
    img_D = imread(image_path[3])
    ref_A = imread(image_path[4])
    ref_B = imread(image_path[5])
    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        img_C = scipy.misc.imresize(img_C, [load_size, load_size])
        img_D = scipy.misc.imresize(img_D, [load_size, load_size])
        ref_A = scipy.misc.imresize(ref_A, [load_size, load_size])
        ref_B = scipy.misc.imresize(ref_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]
        img_C = img_C[h1:h1 + fine_size, w1:w1 + fine_size]
        img_D = img_D[h1:h1 + fine_size, w1:w1 + fine_size]
        ref_A = ref_A[h1:h1 + fine_size, w1:w1 + fine_size]
        ref_B = ref_B[h1:h1 + fine_size, w1:w1 + fine_size]
        if is_noise:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            img_C = np.fliplr(img_C)
            img_D = np.fliplr(img_D)
            ref_A = np.fliplr(ref_A)
            ref_B = np.fliplr(ref_B)

    else:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        img_C = scipy.misc.imresize(img_C, [fine_size, fine_size])
        img_D = scipy.misc.imresize(img_D, [fine_size, fine_size])
        ref_A = scipy.misc.imresize(ref_A, [fine_size, fine_size])
        ref_B = scipy.misc.imresize(ref_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    img_C = img_C / 127.5 - 1.
    img_D = img_D / 127.5 - 1.
    ref_A = ref_A / 127.5 - 1.
    ref_B = ref_B / 127.5 - 1.

    img_AB = np.concatenate((img_A, img_B,img_C,img_D,ref_A,ref_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def resize_and_norm(vgg, img_size):
    for item in ["e2", "e3", "e4"]:
        vgg[item] = tf.image.resize_images(vgg[item], [img_size, img_size])
        vgg[item] = tf.reduce_mean(vgg[item], axis=3, keepdims=True)
        item_max = tf.reduce_max(vgg[item])
        item_min = tf.reduce_min(vgg[item])
        vgg[item] = (vgg[item] - item_min) / (item_max - item_min)
        vgg[item] = vgg[item] * 2.0 - 1
    return tf.concat([vgg["e2"],vgg["e3"],vgg["e4"]],axis=3)


def reconstruction_unet(image, options, reuse=False, name="reconstruction"):
    net={}
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        net["e2"]=e2
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*2, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        net["e3"] = e3
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*4, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        net["e4"] = e4
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*4, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*4, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*4, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*4, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*4, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*4, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*4, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*4, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*2, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)
        pred=tf.nn.tanh(d8)
        return pred,net


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim*2, 7, 1, padding='VALID', name='g_pred_c'))

        return pred[:,:,:,:options.output_c_dim],pred[:,:,:,options.output_c_dim:options.output_c_dim*2]


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.concat([input_layer,noise],axis=3)


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.n_d = args.n_d

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_resnet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf//args.n_d, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.input_c_dim *6],name='real_images')
        self.rec_weight = tf.placeholder(tf.float32,[self.batch_size],name="rec_weight")
        self.noise = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size,self.input_c_dim *2], name="noise")

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim*2]
        self.real_C1 = self.real_data[:, :, :, self.input_c_dim*2:self.input_c_dim*3]
        self.real_C2 = self.real_data[:, :, :, self.input_c_dim*3:self.input_c_dim*4]
        self.ref_A = self.real_data[:, :, :, self.input_c_dim * 4:self.input_c_dim * 5]
        self.ref_B = self.real_data[:, :, :, self.input_c_dim * 5:self.input_c_dim * 6]

        self.noise_A = self.noise[:, :, :, :self.input_c_dim]
        self.noise_B = self.noise[:, :, :, self.input_c_dim:self.input_c_dim * 2]

        self.rec_A,self.net_A = reconstruction_unet(self.real_A, self.options, False, name="reconstruction")
        self.rec_B, self.net_B = reconstruction_unet(self.real_B, self.options, True, name="reconstruction")
        self.rec_refA, self.net_refA = reconstruction_unet(self.ref_A, self.options, True, name="reconstruction")
        self.rec_refB, self.net_refB = reconstruction_unet(self.ref_B, self.options, True, name="reconstruction")

        self.reconstruction_loss = (self.L1_lambda * abs_criterion(self.real_A,self.rec_A) + self.L1_lambda * abs_criterion(self.real_B,self.rec_B)
                                    + self.L1_lambda * abs_criterion(self.ref_A,self.rec_refA)+ self.L1_lambda * abs_criterion(self.ref_B,self.rec_refB))*0.25

        self.net_A = resize_and_norm(self.net_A,self.image_size)
        self.net_B = resize_and_norm(self.net_B, self.image_size)
        self.net_refA = resize_and_norm(self.net_refA, self.image_size)
        self.net_refB = resize_and_norm(self.net_refB, self.image_size)

        self.fake_B , _ = self.generator(tf.concat([self.real_A,self.real_C1,self.net_B],axis=3), self.options, False, name="generatorA2B")
        self.fake_A_, self.rec_C1 = self.generator(tf.concat([self.fake_B,self.noise_B,self.net_A],axis=3), self.options, False, name="generatorB2A")
        self.wrong_recA,self.wrong_recC1 = self.generator(tf.concat([self.fake_B,self.noise_B,self.net_refA],axis=3),self.options, True, name="generatorB2A")

        self.fake_A,_ = self.generator(tf.concat([self.real_B,self.real_C2,self.net_A],axis=3), self.options, True, name="generatorB2A")
        self.fake_B_,self.rec_C2 = self.generator(tf.concat([self.fake_A,self.noise_A,self.net_B],axis=3), self.options, True, name="generatorA2B")
        self.wrong_recB,self.wrong_recC2 = self.generator(tf.concat([self.fake_A,self.noise_A,self.net_refB],axis=3), self.options, True, name="generatorA2B")

        self.GAN_loss_total=0.0
        for i in range(self.n_d):
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name=str(i)+"_discriminatorB")
            self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name=str(i)+"_discriminatorA")
            self.DB_fake_wrong = self.discriminator(self.wrong_recB, self.options, reuse=True, name=str(i) + "_discriminatorB")
            self.DA_fake_wrong = self.discriminator(self.wrong_recA, self.options, reuse=True, name=str(i) + "_discriminatorA")
            self.GAN_loss_total += (self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))
                                    +self.criterionGAN(self.DB_fake_wrong, tf.ones_like(self.DB_fake_wrong)) + self.criterionGAN(self.DA_fake_wrong, tf.ones_like(self.DA_fake_wrong)))

        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_C1, self.rec_C1) * tf.reduce_mean(self.rec_weight)\
            + self.L1_lambda * abs_criterion(self.real_C2, self.rec_C2) * tf.reduce_mean(self.rec_weight)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_C1, self.rec_C1) * tf.reduce_mean(self.rec_weight) \
            + self.L1_lambda * abs_criterion(self.real_C2, self.rec_C2) * tf.reduce_mean(self.rec_weight)
        self.g_loss = self.GAN_loss_total + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + self.L1_lambda * abs_criterion(self.real_C1, self.rec_C1) * tf.reduce_mean(self.rec_weight) \
            + self.L1_lambda * abs_criterion(self.real_C2, self.rec_C2) * tf.reduce_mean(self.rec_weight)\
            + self.L1_lambda * abs_criterion(self.wrong_recC1, self.noise_B) + self.L1_lambda * abs_criterion(self.wrong_recC2,self.noise_A)

        self.g_rec_A = abs_criterion(self.real_A, self.fake_A_)
        self.g_rec_B = abs_criterion(self.real_B, self.fake_B_)
        self.g_rec_C1 = abs_criterion(self.real_C1, self.rec_C1) * tf.reduce_mean(self.rec_weight)
        self.g_rec_C2 = abs_criterion(self.real_C2, self.rec_C2) * tf.reduce_mean(self.rec_weight)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')

        self.fake_A_sample_wrong = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample_wrong')
        self.fake_B_sample_wrong = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample_wrong')
        self.d_loss_item=[]
        for i in range(self.n_d):
            self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name=str(i)+"_discriminatorB")
            self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name=str(i)+"_discriminatorA")
            self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name=str(i)+"_discriminatorB")
            self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name=str(i)+"_discriminatorA")

            self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
            self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
            self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
            self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
            self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
            self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
            self.d_loss = self.da_loss + self.db_loss

            self.DB_real_wrong = self.discriminator(self.ref_B, self.options, reuse=True, name=str(i) + "_discriminatorB")
            self.DA_real_wrong = self.discriminator(self.ref_A, self.options, reuse=True, name=str(i) + "_discriminatorA")
            self.DB_fake_sample_wrong = self.discriminator(self.fake_B_sample_wrong, self.options, reuse=True,
                                                     name=str(i) + "_discriminatorB")
            self.DA_fake_sample_wrong = self.discriminator(self.fake_A_sample_wrong, self.options, reuse=True,
                                                     name=str(i) + "_discriminatorA")

            self.db_loss_real_wrong = self.criterionGAN(self.DB_real_wrong, tf.ones_like(self.DB_real_wrong))
            self.db_loss_fake_wrong = self.criterionGAN(self.DB_fake_sample_wrong, tf.zeros_like(self.DB_fake_sample_wrong))
            self.db_loss_wrong = (self.db_loss_real_wrong + self.db_loss_fake_wrong) / 2
            self.da_loss_real_wrong = self.criterionGAN(self.DA_real_wrong, tf.ones_like(self.DA_real_wrong))
            self.da_loss_fake_wrong = self.criterionGAN(self.DA_fake_sample_wrong, tf.zeros_like(self.DA_fake_sample_wrong))
            self.da_loss_wrong = (self.da_loss_real_wrong + self.da_loss_fake_wrong) / 2
            self.d_loss_wrong = self.da_loss_wrong + self.db_loss_wrong


            self.d_loss_item.append((self.d_loss+self.d_loss_wrong)*0.5)

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_rec_A_sum = tf.summary.scalar("g_rec_A", self.g_rec_A)
        self.g_rec_B_sum = tf.summary.scalar("g_rec_B", self.g_rec_B)
        self.g_rec_C1_sum = tf.summary.scalar("g_rec_C1", self.g_rec_C1)
        self.g_rec_C2_sum = tf.summary.scalar("g_rec_C2", self.g_rec_C2)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.GAN_loss_total_sum = tf.summary.scalar("GAN_loss_total", self.GAN_loss_total)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum,self.GAN_loss_total_sum,
                                       self.g_rec_A_sum,self.g_rec_B_sum,self.g_rec_C1_sum,self.g_rec_C2_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.r_sum_A = tf.summary.scalar("reconstruct_A", self.L1_lambda * abs_criterion(self.real_A,self.rec_A))
        self.r_sum_B = tf.summary.scalar("reconstruct_B", self.L1_lambda * abs_criterion(self.real_B, self.rec_B))
        self.r_sum = tf.summary.merge([self.r_sum_A,self.r_sum_B])

        self.test_A_merge = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*4,self.input_c_dim], name='test_A_merge')
        self.test_B_merge = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size * 4, self.input_c_dim],name='test_B_merge')

        self.test_A = self.test_A_merge[:,:,:self.image_size,:]
        self.test_mA = self.test_A_merge[:, :, self.image_size:self.image_size*2, :]
        self.test_refB = self.test_A_merge[:, :, self.image_size*2:self.image_size * 3, :]
        self.test_recA = self.test_A_merge[:, :, self.image_size * 3:self.image_size * 4, :]

        self.test_B = self.test_B_merge[:, :, :self.image_size, :]
        self.test_mB = self.test_B_merge[:, :, self.image_size:self.image_size * 2, :]
        self.test_refA = self.test_B_merge[:, :, self.image_size * 2:self.image_size * 3, :]
        self.test_recB = self.test_B_merge[:, :, self.image_size * 3:self.image_size * 4, :]


        _,self.test_refA_net = reconstruction_unet(self.test_refA, self.options, True, name="reconstruction")
        _, self.test_refB_net = reconstruction_unet(self.test_refB, self.options, True, name="reconstruction")
        _, self.test_recA_net = reconstruction_unet(self.test_recA, self.options, True, name="reconstruction")
        _, self.test_reCB_net = reconstruction_unet(self.test_recB, self.options, True, name="reconstruction")

        self.test_refA_net=resize_and_norm(self.test_refA_net,self.image_size)
        self.test_refB_net = resize_and_norm(self.test_refB_net,self.image_size)
        self.test_recA_net = resize_and_norm(self.test_recA_net, self.image_size)
        self.test_reCB_net = resize_and_norm(self.test_reCB_net, self.image_size)

        self.testB,_ = self.generator(tf.concat([self.test_A,self.test_mA,self.test_refB_net],axis=3), self.options, True, name="generatorA2B")
        self.rec_testA,self.rec_mA = self.generator(tf.concat([self.testB,self.noise_B,self.test_recA_net],axis=3), self.options, True, name="generatorB2A")
        self.testA,_ = self.generator(tf.concat([self.test_B,self.test_mB,self.test_refA_net],axis=3), self.options, True, name="generatorB2A")
        self.rec_testB, self.rec_mB = self.generator(tf.concat([self.testA,self.noise_A,self.test_reCB_net],axis=3), self.options, True, name="generatorA2B")


        t_vars = tf.trainable_variables()
        self.d_vars_item=[]
        for i in range(self.n_d):
            d_vars = [var for var in t_vars if str(i)+'_discriminator' in var.name]
            self.d_vars_item.append(d_vars)
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.r_vars = [var for var in t_vars if 'reconstruction' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim_item=[]
        for i in range(self.n_d):
            d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item[i], var_list=self.d_vars_item[i])
            self.d_optim_item.append(d_optim)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        self.r_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.reconstruction_loss, var_list=self.r_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir,"./logs"),self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainC'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            np.random.shuffle(dataC)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                index_C = np.random.randint(0,len(dataC))
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataC[index_C * self.batch_size:(index_C + 1) * self.batch_size],
                                       dataC[(len(dataC)-1-index_C) * self.batch_size:(len(dataC)-index_C) * self.batch_size],
                                       dataA[(len(dataA)-1-idx) * self.batch_size:((len(dataA)-idx)) * self.batch_size],
                                       dataB[(len(dataB)-1-idx) * self.batch_size:((len(dataB)-idx)) * self.batch_size]
                                       ))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size,is_testing=False,is_noise=False) for batch_file in batch_files]

                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                noise=np.random.uniform(-1.0,1.0,[self.batch_size,self.image_size,self.image_size,self.input_c_dim*2])

                _,rec_loss, summary_str = self.sess.run(
                    [self.r_optim,self.reconstruction_loss,self.r_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr, self.rec_weight: [1.0], self.noise: noise})
                self.writer.add_summary(summary_str, counter)

                fake_A, fake_B,wrong_fakeA,wrong_fakeB, _, summary_str,g_loss,a2b,b2a,g_rec_A,g_rec_B,g_rec_C1,g_rec_C2,GAN_loss = self.sess.run(
                    [self.fake_A, self.fake_B,self.wrong_recA,self.wrong_recB,self.g_optim, self.g_sum,self.g_loss,self.g_loss_a2b,self.g_loss_b2a,self.g_rec_A,self.g_rec_B,self.g_rec_C1,self.g_rec_C2,self.GAN_loss_total],
                    feed_dict={self.real_data: batch_images, self.lr: lr,self.rec_weight:[1.0],self.noise:noise})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                [wrong_fakeA, wrong_fakeB] = self.pool([wrong_fakeA, wrong_fakeB])

                # Update D network
                da_loss=None
                db_loss=None
                for i in range(self.n_d):
                    _, summary_str,da_loss,db_loss = self.sess.run(
                        [self.d_optim_item[i], self.d_sum,self.da_loss,self.db_loss],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.fake_A_sample_wrong:wrong_fakeA,
                                   self.fake_B_sample_wrong:wrong_fakeB,
                                   self.lr: lr,self.rec_weight:[1.0],self.noise:noise})
                self.writer.add_summary(summary_str, counter)
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss %4.4f GAN %4.4f recon_loss %4.4f a2b %4.4f b2a %4.4f da %4.4f db %4.4f A: %4.4f B: %4.4f C1: %4.4f C2: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, g_loss,GAN_loss,rec_loss, a2b, b2a, da_loss, db_loss,g_rec_A,g_rec_B,g_rec_C1,g_rec_C2)))


                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testC'))
        noise = np.random.uniform(-1.0, 1.0, [self.batch_size, self.image_size, self.image_size, self.input_c_dim * 2])
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        np.random.shuffle(dataC)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size],
                               dataC[:self.batch_size], dataC[self.batch_size:self.batch_size+1],
                               dataA[self.batch_size:self.batch_size+1], dataB[self.batch_size:self.batch_size+1]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        real_A,real_B,real_C1,real_C2,fake_A, fake_B,fake_A_,fake_B_,rec_C1,rec_C2,rec_A,rec_B,ref_A,ref_B,wrong_recA,wrong_recB,wrong_recC1,wrong_recC2= self.sess.run(
            [self.real_A,self.real_B,self.real_C1,self.real_C2,self.fake_A, self.fake_B,self.fake_A_,self.fake_B_,self.rec_C1,self.rec_C2,self.rec_A,self.rec_B,
             self.ref_A,self.ref_B,self.wrong_recA,self.wrong_recB,self.wrong_recC1,self.wrong_recC2],
            feed_dict={self.real_data: sample_images,self.rec_weight:[1.0],self.noise:noise}
        )
        merge_A = np.concatenate([real_A,real_C1,real_B,fake_B,fake_A_,rec_C1,rec_A,ref_A,wrong_recA,wrong_recC1],axis=2)
        merge_B = np.concatenate([real_B, real_C2,real_A, fake_A, fake_B_, rec_C2, rec_B,ref_B,wrong_recB,wrong_recC2], axis=2)
        save_images(merge_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(merge_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/point_testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/point_testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        noise = np.random.uniform(-1.0, 1.0, [self.batch_size, self.image_size, self.image_size, self.input_c_dim * 2])
        out_var,rec_var,rec_m,in_var = (self.testB,self.rec_testA,self.rec_mA,self.test_A_merge) if args.which_direction == 'AtoB' else (
            self.testA,self.rec_testB,self.rec_mB, self.test_B_merge)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img,rec_img,rec_m_img = self.sess.run([out_var,rec_var,rec_m], feed_dict={in_var: sample_image,self.noise:noise})
            ipt_img = sample_image[:,:,:self.image_size,:]
            message_img = sample_image[:, :, self.image_size:self.image_size*2, :]
            disguise_img = sample_image[:,:,self.image_size*2:self.image_size*3,:]
            rec_ref = sample_image[:,:,self.image_size*3:self.image_size*4,:]
            merge_img= np.concatenate([ipt_img,message_img,disguise_img,fake_img,rec_ref,rec_img,rec_m_img],axis=2)
            save_images(merge_img, [1, 1], image_path)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--n_d', dest='n_d', type=int, default=4, help='# how many branches of discriminator')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    tf.app.run()

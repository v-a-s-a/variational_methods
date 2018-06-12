#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import tensorflow as tf
import tensorflow_probability as tfp

from absl import flags
import numpy as np
import datetime


tfd = tf.contrib.distributions

## Command line interface

# Model specification
flags.DEFINE_integer('N',
    default=0,
    help='Sample size.')
flags.DEFINE_integer('M',
    default=0,
    help='Number of sites.')
flags.DEFINE_integer('D',
    default=0,
    help='Latent dimension.')

# Training
flags.DEFINE_integer('epochs',
    default=100,
    help='Training epochs.')

# Runtime
flags.DEFINE_string('plink_tfrecords',
    default='',
    help='File path for the genotype in .tfrecords format.')
flags.DEFINE_string('log_dir',
    default='./logs/',
    help='Directory for tensorboard logs.')


FLAGS = flags.FLAGS


## Helpers for model construction
def make_encoder(data, sample_size, site_num, latent_dim):
    data = tf.reshape(data, [sample_size, site_num])

    # sample latent variables
    x = tf.layers.dense(inputs=data,
            units=512, activation=tf.nn.sigmoid)
    x = tf.layers.dense(inputs=x,
            units=256, activation=tf.nn.sigmoid)
    x = tf.layers.dense(inputs=x,
            units=128, activation=tf.nn.sigmoid)
    u_net = tf.layers.dense(inputs=x,
                      units = latent_dim * 2,
                      activation=None)
    u_loc = u_net[..., :latent_dim]
    u_scale = tf.nn.softplus(u_net[..., latent_dim:])
    u = tfd.MultivariateNormalDiag(u_loc, scale_diag=u_scale,
                                   name='sample_latent_U')
    
    # observation latent variables
    x_t = tf.transpose(data)
    x_t = tf.layers.dense(inputs=x_t,
            units=64, activation=tf.nn.sigmoid)
    x_t = tf.layers.dense(inputs=x_t,
            units=32, activation=tf.nn.sigmoid)
    x_t = tf.layers.dense(inputs=x_t,
            units=16, activation=tf.nn.sigmoid)
    v_net = tf.layers.dense(inputs=x_t,
                      units = latent_dim * 2,
                      activation=None)
    v_loc = v_net[..., lantent_dim:]    
    v_scale = tf.nn.softplus(v_net[..., :latent_dim])
    
    v = tfd.MultivariateNormalDiag(v_loc, scale_diag=v_scale,
                                   name='observation_latent_V')
    
    return u, v


def make_conv_encoder(data, sample_size, site_num, latent_dim):
    # sample latent variables
    x = tf.reshape(data, [sample_size, site_num, 1])
    x = tf.layers.conv1d(x, 64, 3, strides=3, padding='SAME', activation=tf.nn.sigmoid)

    x = tf.layers.conv1d(x, 32, 3, strides=3, padding='SAME', activation=tf.nn.sigmoid)

    x = tf.layers.conv1d(x, 16, 3, strides=3, padding='VALID', activation=tf.nn.sigmoid)
    
    x = tf.reshape(x, [sample_size, -1]) # drop channel dimension
    
    u_net = tf.layers.dense(inputs=x,
                      units = latent_dim * 2,
                      activation=None)
    u_loc = u_net[..., :latent_dim]
    u_scale = tf.nn.softplus(u_net[..., latent_dim:])
    u = tfd.MultivariateNormalDiag(u_loc, scale_diag=u_scale,
                                   name='sample_latent_U')

    # site latent variables
    x_t = tf.transpose(data)
    x_t = tf.reshape(x_t, [site_num, sample_size, 1])
    x_t = tf.layers.conv1d(x_t, 64, 3, strides=3, padding='SAME', activation=tf.nn.sigmoid)

    x_t = tf.layers.conv1d(x_t, 32, 3, strides=3, padding='SAME', activation=tf.nn.sigmoid)

    x_t = tf.layers.conv1d(x_t, 16, 3, strides=3, padding='VALID', activation=tf.nn.sigmoid)
    
    x_t = tf.reshape(x_t, [site_num, -1]) # drop channel dimension
    
    v_net = tf.layers.dense(inputs=x_t,
                      units = latent_dim * 2,
                      activation=None)
    v_loc = v_net[..., :latent_dim]
    v_scale = tf.nn.softplus(v_net[..., latent_dim:])
    v = tfd.MultivariateNormalDiag(v_loc, scale_diag=v_scale,
                                   name='site_latent_V')
    
    return u, v


def make_conv_decoder(u , v, sample_size, site_num, latent_dim):
    
    ## convolutional decoder
    z = tf.concat([u, v], 0)
    print(z.shape)
    x = tf.reshape(z, [sample_size + site_num, 1, latent_dim, 1])
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=(1, 3),
        strides=(1, 3), padding='SAME', activation=tf.nn.sigmoid)
    x = tf.squeeze(x, [1])
    padding = tf.constant([[0, 0], [1, 0], [0, 0]])
    x = tf.pad(x, padding)
    print(x.shape)

    x = tf.expand_dims(x, axis=1)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=(1, 4),
        strides=(1, 4), padding='SAME', activation=tf.nn.sigmoid)
    x = tf.squeeze(x, [1])
    print(x.shape)

    x = tf.expand_dims(x, axis=1)
    x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=(1, 3),
        strides=(1, 3), padding='VALID', activation=tf.nn.sigmoid)
    x = tf.squeeze(x, [1])
    x = tf.pad(x, padding)
    print(x.shape)

    decoder_net = x

    # decoder_net = tf.layers.dense(inputs=x,
    #     units=sample_size * site_num, activation=None)

    # ## dot product decoder
    # z = tf.tensordot(u, v, axes=[[1], [1]])
    # decoder_net = tf.reshape(z, [1, site_num*sample_size])

    return tfd.Independent(tfd.Binomial(logits=decoder_net,
                            total_count=2.0),
                            reinterpreted_batch_ndims=1,
                            name="decoder_distribution")


def make_decoder(u, v, sample_size, site_num, latent_dim):
    
    # "dot product decoder"
    # z = tf.tensordot(u, v, axes=[[1], [1]])
    # z = tf.reshape(z, [1, sample_size * site_num])
    
    # "neural network decoder"
    z = tf.concat([u, v], 1)
    z = tf.layers.dense(inputs=z,
            units=128, activation=tf.nn.sigmoid)
    z = tf.layers.dense(inputs=z,
             units=256, activation=tf.nn.sigmoid)
    z = tf.layers.dense(inputs=z,
            units=512, activation=tf.nn.sigmoid)
    z = tf.layers.dense(inputs=z,
            units=sample_size * site_num, activation=None)
    
    # assume fixed, unit variance
    data_dist = tfd.Independent(tfd.Normal(loc=z, scale=1.,
                            name='posterior_p'))
        
    return data_dist


def make_prior(latent_dim):
    u_prior =  tfd.MultivariateNormalDiag(scale_diag=tf.ones(latent_dim),
                                    name='U')
    v_prior = tfd.MultivariateNormalDiag(scale_diag=tf.ones(latent_dim),
                                    name='V')

    return u_prior, v_prior



def main(argv):
    graph = tf.Graph()
    with graph.as_default():

        # input pipeline
        X = np.random.binomial(2, 0.5, size=(FLAGS.N, FLAGS.M))
        X = X.astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.batch(FLAGS.N)
        iterator = dataset.make_initializable_iterator()
        data = iterator.get_next()
        
        with tf.variable_scope('priors'):
            u_prior, v_prior = make_prior(latent_dim=FLAGS.D)
            
        # inference network; encoder
        with tf.variable_scope('encoder'):
            u_encoder, v_encoder = make_conv_encoder(data,
                                        latent_dim=FLAGS.D,
                                        sample_size=FLAGS.N,
                                        site_num=FLAGS.M)
        
        u = u_encoder.sample()
        v = v_encoder.sample()

        # generative network; decoder
        with tf.variable_scope('decoder'):
            decoder_p = make_conv_decoder(u, v, latent_dim=FLAGS.D, site_num=FLAGS.M,
                                    sample_size=FLAGS.N)

        # loss
        u_kl = tf.reduce_sum(tfd.kl_divergence(u_encoder, u_prior))
        v_kl = tf.reduce_sum(tfd.kl_divergence(v_encoder, v_prior))
        likelihood = tf.reduce_sum(decoder_p.log_prob(tf.reshape(data, [1, FLAGS.N*FLAGS.M])))
        elbo = -u_kl - v_kl + likelihood
        tf.summary.scalar('elbo', elbo)
        tf.summary.scalar('minus_u_kl', tf.negative(u_kl))
        tf.summary.scalar('minus_v_kl', tf.negative(v_kl))
        tf.summary.scalar('likelihood', likelihood)
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(0.001).minimize(-elbo)
        merged = tf.summary.merge_all()

    # training
    with tf.variable_scope('training'):
        run = 'run-{date:%d.%m.%Y_%H:%M:%S}'.format(date=datetime.datetime.now())
        tb_writer = tf.summary.FileWriter(FLAGS.log_dir + run, graph=graph)

        with tf.Session(graph=graph) as sess:    
            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.epochs):
                sess.run(iterator.initializer)
                while True:
                    try:
                        _, tb_summary, epoch_elbo = sess.run([optimizer, merged, elbo])
                        print(epoch_elbo)
                        tb_writer.add_summary(tb_summary, epoch)
                    except tf.errors.OutOfRangeError:
                        break


if __name__ == '__main__':
    tf.app.run()
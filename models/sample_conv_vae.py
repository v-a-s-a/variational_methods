#!/usr/bin/env python
'''
A convolutional variational autoencoder. This is used to test whether
we can simulate samples that have principal components "similar" to what
we observe in real data.

The decoder/encoder networks are tailored to our `large_test_indep` dataset
that consists of 127,464 LD-pruned sites measured in 7,763 samples.
'''

import warnings
warnings.simplefilter('always', DeprecationWarning)

import tensorflow as tf
import tensorflow_probability as tfp


from absl import flags
import numpy as np
import datetime

tfd = tf.contrib.distributions


## Command line interface
# Model specification
flags.DEFINE_integer('N',
    default=7763,
    help='Sample size.')
flags.DEFINE_integer('M',
    default=127464,
    help='Number of sites.')
flags.DEFINE_integer('D',
    default=20,
    help='Latent dimension.')
flags.DEFINE_integer('sim_batches',
    default=10,
    help='Number of batches (of batch_size specified separately) to simulate genotypes for.')
# Training
flags.DEFINE_integer('epochs',
    default=100,
    help='Training epochs.')
flags.DEFINE_integer('batch_size',
    default=100,
    help='Training batch size for stochastic gradient descent.')

# Runtime
flags.DEFINE_string('plink_tfrecords',
    default='',
    help='File path for the genotype in .tfrecords format.')
flags.DEFINE_string('log_dir',
    default='./logs/',
    help='Directory for tensorboard logs.')

FLAGS = flags.FLAGS


# encoder/decoder specification
def make_conv_encoder(data, batch_size, num_features, latent_dimension=2):
    x = tf.reshape(data, [batch_size, num_features, 1])
    print('Encoder layer1 input: {}'.format(x.shape))
    x = tf.layers.conv1d(x, 64, 3, strides=3, padding='SAME', activation=tf.nn.relu)

    print('Encoder layer2 input: {}'.format(x.shape))
    x = tf.layers.conv1d(x, 32, 3, strides=3, padding='SAME', activation=tf.nn.relu)

    print('Encoder layer3 input: {}'.format(x.shape))
    x = tf.layers.conv1d(x, 16, 3, strides=3, padding='SAME', activation=tf.nn.relu)

    print('Encoder layer4 input: {}'.format(x.shape))
    x = tf.layers.conv1d(x, 8, 3, strides=3, padding='VALID', activation=tf.nn.relu)  
    x = tf.reshape(x, [batch_size, -1]) # drop channel dimension
    
    print('Encoder final layer input: {}'.format(x.shape))
    encoder_net = tf.layers.dense(x, units=latent_dimension*2, activation=tf.nn.sigmoid)
    
    loc = encoder_net[..., :latent_dimension]
    scale = tf.nn.softplus(encoder_net[..., latent_dimension:] + 0.5)

    return tfd.MultivariateNormalDiag(loc=loc,
            scale_diag=scale,
            name="encoder_distribution")


def make_conv_decoder(latent_code, batch_size, num_features, latent_dimension=2):
    latent_code = tf.squeeze(latent_code)

    print('Decoder matching-layer input: {}'.format(latent_code.shape))
    x = tf.layers.dense(inputs=latent_code, units=1632)

    print('Decoder layer1 input: {}'.format(x.shape))
    x = tf.reshape(x, [batch_size, 1, 1632, 1])
    x = tf.layers.conv2d_transpose(x, filters=16, kernel_size=(1, 3),
        strides=(1, 3), padding='VALID', activation=tf.nn.relu)
    x = tf.squeeze(x, [1])
    padding = tf.constant([[0, 0], [1, 0], [0, 0]])
    x = tf.pad(x, padding)

    print('Decoder layer2 input: {}'.format(x.shape))
    x = tf.expand_dims(x, axis=1)
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=(1, 3),
        strides=(1, 3), padding='SAME', activation=tf.nn.relu)
    x = tf.squeeze(x, [1])
    
    print('Decoder layer3 input: {}'.format(x.shape))
    x = tf.expand_dims(x, axis=1)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=(1, 3),
        strides=(1, 3), padding='SAME', activation=tf.nn.relu)
    x = tf.squeeze(x, [1])
    x = tf.pad(x, padding)

    print('Decoder layer4 input: {}'.format(x.shape))
    x = tf.expand_dims(x, axis=1)
    x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=(1, 3),
        strides=(1, 3), padding='SAME', activation=tf.nn.sigmoid)
    x = tf.squeeze(x, [1, 3])
    
    print('Decoder final layer input: {}'.format(x.shape))

    decoder_net = tf.slice(x, [0, 0], [batch_size, num_features])
    
    return tfd.Independent(tfd.Binomial(logits=decoder_net,
                            total_count=2.0),
                            reinterpreted_batch_ndims=1,
                            name="decoder_distribution")


def make_prior(latent_dimension):
    prior =  tfd.MultivariateNormalDiag(scale_diag=tf.ones(latent_dimension),
                                    name="prior_distribution")
    return prior


def decode_tfrecords(tfrecords_filename, m_variants):
    '''
    Parse a tf.string pointing to *.tfrecords into a genotype tensor,  rows: variants, cols: samples)
    Helpful blog post:
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    '''
    data = tf.parse_example([tfrecords_filename],
        {'genotypes': tf.FixedLenFeature([], tf.string)})

    gene_vector = tf.decode_raw(data['genotypes'], tf.int8)
    gene_vector = tf.reshape(gene_vector, [1, m_variants])

    return gene_vector



def main(argv):
    graph = tf.Graph()
    with graph.as_default():

        # input pipeline
        dataset = tf.data.TFRecordDataset(FLAGS.plink_tfrecords, compression_type=tf.constant('ZLIB'))
        dataset = dataset.map(lambda fn: decode_tfrecords(fn, FLAGS.M))
        dataset = dataset.batch(FLAGS.batch_size)
        iterator = dataset.make_initializable_iterator()
        data = iterator.get_next()
        data = tf.cast(data, tf.float32)
        
        # inference network; encoder
        with tf.variable_scope('encoder'):
            encoder_q = make_conv_encoder(data, batch_size=FLAGS.batch_size,
                                        latent_dimension=FLAGS.D,
                                        num_features=FLAGS.M)
            z = encoder_q.sample()

        # generative network; decoder
        with tf.variable_scope('decoder'):
            decoder_p = make_conv_decoder(z, batch_size=FLAGS.batch_size,
                                            latent_dimension=FLAGS.D, num_features=FLAGS.M)
            simulated_geno = tf.concat(tf.map_fn(lambda x: decoder_p.sample(), tf.range(FLAGS.sim_batches)), 0)
        
        # prior
        with tf.variable_scope('prior'):
            prior = make_prior(latent_dimension=d)

        # loss
        likelihood = decoder_p.log_prob(data)
        elbo = tf.reduce_mean(tfd.kl_divergence(encoder_q, prior) - likelihood)

        # optimizer
        optimizer = tf.train.AdamOptimizer(0.001).minimize(elbo)

        # inference routine
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            conv_elbo_record = list()
            for epoch in range(FLAGS.epochs):
                sess.run(iterator.initializer)
                
                conv_sample_latent_codes = list()
                while True:
                    try:
                        _, epoch_elbo, epoch_latent_code, sim_geno = sess.run([optimizer, elbo, latent_code, sim_geno])
                        conv_sample_latent_codes.append(epoch_latent_code)
                        print('EPOCH {epoch}: ELBO {epoch_elbo}'.format(epoch=epoch, epoch_elbo=epoch_elbo))
                        np.save('sim_geno.' + epoch + '.npy', sim_geno)
                    except tf.errors.OutOfRangeError:
                        conv_elbo_record.append(epoch_elbo)
                        break


if __name__ == '__main__':
    tf.app.run()

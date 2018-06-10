#!/usr/bin/env python

import tensorflow as tf
from pandas_plink import read_plink
import pandas as pd
import numpy as np
import argparse as arg


def __main__(plink_file, tfrecords_file, tf_opts):
    bim, fam, G = read_plink(plink_file)
    G = np.array(G.T, dtype=np.int8)
    G[np.isnan(G)] = 0
    N = G.shape[0]
    M = G.shape[1]

    def write_record(row, writer_handle):
        '''
        row: a sample's genotype vector.
        '''
        # wrap raw byte values
        genotypes_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[row.tostring()]))

        # convert to Example
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'genotypes': genotypes_feature}))

        writer_handle.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter(tfrecords_file, options=tf_opts) as tfwriter:
        np.apply_along_axis(write_record, axis=1, arr=G, writer_handle=tfwriter)


if __name__ == '__main__':
    tf_opts =  tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    parser = arg.ArgumentParser()
    parser.add_argument('--plink-file', dest='plink_file', action='store')
    parser.add_argument('--tfrecords-file', dest='tfrecords', action='store')
    args = parser.parse_args()

    __main__(args.plink_file, args.tfrecords, tf_opts)
    
#!/usr/bin/python

import cPickle as pickle
import itertools
import os
import tensorflow as tf

TRAIN_FILE_NAME = 'cifar10-train.dat'
TEST_FILE_NAME = 'cifar10-test.dat'
TRAIN_DATA_FILES = [
   'data_batch_1',
   'data_batch_2',
   'data_batch_3',
   'data_batch_4',
   'data_batch_5',
]

TEST_DATA_FILES = [ 'test_batch' ]

DATA_DIR = 'cifar-10-batches-py'

def process_batches(dest_file_name, src_file_names):
  writer = tf.python_io.TFRecordWriter(dest_file_name)
  try:
    for src_file_name in src_file_names:
      with open(os.path.join(DATA_DIR,  src_file_name), 'rb') as f:
        src_dict = pickle.load(f)
      for label, img_data in itertools.izip(
            src_dict['labels'], src_dict['data']):
        example = tf.train.Example(features=tf.train.Features(feature={
          'image_raw': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[img_data.tostring()])),
          'label': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())
  finally:
    writer.close()

def main():
  process_batches(TRAIN_FILE_NAME, TRAIN_DATA_FILES)
  process_batches(TEST_FILE_NAME, TEST_DATA_FILES)

if __name__ == '__main__':
  main()

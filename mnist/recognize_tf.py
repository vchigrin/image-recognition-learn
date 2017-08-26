#!/usr/bin/python -Ou

import collections
import copy
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import itertools
import math
import numpy as np
import scipy.signal
import sys
import time
import tensorflow as tf

N_L0_UNITS = 75
N_L1_UNITS = 25
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
KERNEL_SIZE = 4
KERNELS_COUNT = 16
POOLING_SIZE = 2
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
LEARN_RATE = 0.0001
WEIGHT_DECAY_RATE = 0.04
# How much use old, history date, relative to current batch gradients.
RMSPROP_DECAY_RATE = 0.9
N_EPOCHS = 25
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.1

FUNCTION_MODE = None

def load_csv(file_name, has_label):
  data_arrays = []
  label_arrays = []
  with open(file_name, 'r') as f:
    header = f.readline()
    for line in f:
      parts = [int(p) for p in line.strip().split(',')]
      if has_label:
        data = np.array(parts[1:], dtype=np.float64)
        label_arrays.append(parts[0])
      else:
        data = np.array(parts, dtype=np.float64)
      data /= 255.
      data_arrays.append(data)
  return data_arrays, label_arrays


def prepare_input(input_batches):
  new_label_arrays = []
  new_arrays = []
  for label, pixels in itertools.izip(*input_batches):
    labels_matrix = np.zeros(shape=( N_OUTPUT_UNITS), dtype=np.float32)
    labels_matrix[label] = 1
    new_label_arrays.append(labels_matrix)
    new_arrays.append(pixels.reshape(INPUT_HEIGHT, INPUT_WIDTH, 1))
  return (new_label_arrays, new_arrays)


def get_data_streams():
  dataset = fuel.datasets.H5PYDataset(
      'kaggle-mnist.hdf5', which_sets=('train',))
  print 'Data loaded. Total examples {}'.format(
       dataset.num_examples)
  num_train_examples = int(dataset.num_examples * (1 - VALIDATION_DATA_PART))
  train_scheme = fuel.schemes.SequentialScheme(
      examples=num_train_examples,
      batch_size=BATCH_SIZE)
  validation_scheme = fuel.schemes.SequentialScheme(
      examples=range(num_train_examples, dataset.num_examples),
      batch_size=BATCH_SIZE)
  train_stream = fuel.transformers.Mapping(
      fuel.streams.DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=train_scheme),
      prepare_input)
  validation_stream = fuel.transformers.Mapping(
      fuel.streams.DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=validation_scheme),
      prepare_input)
  return train_stream, validation_stream


def report_statistics(
    stream,
    session,
    cross_entropy_sum, errors_count,
    in_sample, target,
    i_epoch,
    which_set, summary_writer):
  num_errors = 0
  num_examples = 0
  sum_cross_entropy = 0
  for batch_dict in stream.get_epoch_iterator(as_dict=True):
    cur_cross_entropy, cur_errors_count = session.run(
        [cross_entropy_sum, errors_count],
        feed_dict={
          in_sample: batch_dict['pixels'],
          target: batch_dict['labels']})
    num_errors = cur_errors_count
    sum_cross_entropy = cur_cross_entropy
    num_examples += len(batch_dict['labels'])

  print('\n{} set cost {} train stream errors percent {}'.format(
      which_set,
      sum_cross_entropy / num_examples, num_errors / float(num_examples)))
  summary = tf.summary.Summary(value=[
    tf.Summary.Value(
        tag=which_set + "_cross_entropy_mean",
        simple_value = sum_cross_entropy / num_examples),
    tf.Summary.Value(
        tag=which_set + "_errors_percent",
        simple_value = float(num_errors) / num_examples),
  ])
  summary_writer.add_summary(summary, i_epoch)


def main():
  in_sample = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, 1])
  target = tf.placeholder(tf.float32, [None, N_OUTPUT_UNITS])

  conv_kernels = tf.get_variable(
      'conv_kernels',
      shape=(KERNEL_SIZE, KERNEL_SIZE, 1, KERNELS_COUNT))

  kernels_summary = tf.transpose(conv_kernels, perm=[3, 0, 1, 2])
  tf.summary.image(
      'conv_kernels',
      kernels_summary,
      max_outputs=KERNELS_COUNT)
  conv_output = tf.nn.conv2d(
      in_sample,
      conv_kernels,
      strides=[1, 1, 1, 1],
      padding='SAME')
  tf.summary.image(
      'input',
      in_sample)
  tf.summary.image(
      'kernel0_convolved',
      conv_output[:,:,:,:1])

  conv_relu = tf.nn.relu(conv_output)
  pool_output = tf.nn.pool(
      conv_relu,
      window_shape=(KERNEL_SIZE, KERNEL_SIZE),
      strides=(KERNEL_SIZE, KERNEL_SIZE),
      pooling_type='MAX',
      padding='SAME')
  pooled_width = (INPUT_WIDTH + KERNEL_SIZE - 1) / KERNEL_SIZE
  pooled_height = (INPUT_HEIGHT + KERNEL_SIZE - 1) / KERNEL_SIZE
  pooled_size = pooled_width * pooled_height * KERNELS_COUNT
  pooling_lineralized = tf.reshape(
      tensor=pool_output,
      shape=[-1, pooled_size])


  layer0_weights = tf.get_variable('l0_weights', shape=(pooled_size, N_L0_UNITS))
  layer0_biases = tf.get_variable('l0_biases', shape=(1, N_L0_UNITS))
  layer0_output = tf.nn.relu(
      tf.matmul(pooling_lineralized, layer0_weights) + layer0_biases,
      'layer0_output')

  layer1_weights = tf.get_variable('l1_weights', shape=(N_L0_UNITS, N_L1_UNITS))
  layer1_biases = tf.get_variable('l1_biases', shape=(1, N_L1_UNITS))
  layer1_output = tf.nn.relu(
      tf.matmul(layer0_output, layer1_weights) + layer1_biases, 'layer1_output')

  output_weights = tf.get_variable(
      'output_weights', shape=(N_L1_UNITS, N_OUTPUT_UNITS))
  output_biases = tf.get_variable('output_biases', shape=(1, N_OUTPUT_UNITS))
  result = tf.nn.softmax(
      tf.matmul(layer1_output, output_weights) + output_biases,
      name='result')

  cross_entropy = tf.reduce_sum(
      -tf.reduce_sum(target * tf.log(result), reduction_indices=[1]))

  cross_entropy_sum = tf.reduce_sum(cross_entropy)

  regularized_cross_entropy = (cross_entropy_sum +
      (tf.reduce_sum(tf.square(layer0_weights)) * WEIGHT_DECAY_RATE) / 2)

  regularized_cross_entropy = (regularized_cross_entropy +
      (tf.reduce_sum(tf.square(layer1_weights)) * WEIGHT_DECAY_RATE) / 2)

  predictions = tf.argmax(result, axis=1)
  target_predictions = tf.argmax(target, axis=1)
  errors_count = tf.count_nonzero(
      tf.not_equal(target_predictions, predictions))

  train_step = tf.train.RMSPropOptimizer(
      LEARN_RATE, decay=RMSPROP_DECAY_RATE).minimize(regularized_cross_entropy)

  session = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  train_stream, validation_stream = get_data_streams()
  writer = tf.summary.FileWriter("/home/slava/tf_graph", session.graph)

  image_summaries = tf.summary.merge_all()

  index_for_image_stat = 0
  try:
    for i_epoch in xrange(N_EPOCHS):
      print('Epoch {}\n'.format(i_epoch))
      all_batches = list(train_stream.get_epoch_iterator())
      for index, batches in enumerate(all_batches):
        sys.stdout.write('Train Batch {}/{}\r'.format(index, len(all_batches)))
        sys.stdout.flush()
        label_to_batch = dict(zip(train_stream.sources, batches))
        session.run(
            train_step,
            feed_dict={
              in_sample: label_to_batch['pixels'],
              target: label_to_batch['labels']})
        if index_for_image_stat % 100 == 0:
          summary = session.run(
              image_summaries,
              feed_dict={
                in_sample: label_to_batch['pixels']
              })
          writer.add_summary(summary, index_for_image_stat)
        index_for_image_stat += 1

      report_statistics(
          train_stream,
          session,
          cross_entropy_sum, errors_count,
          in_sample, target,
          i_epoch,
          'train', writer)
      report_statistics(
          validation_stream,
          session,
          cross_entropy_sum, errors_count,
          in_sample, target,
          i_epoch,
          'validation', writer)
  finally:
    writer.close()


if __name__ == '__main__':
  main()

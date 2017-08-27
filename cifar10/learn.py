#!/usr/bin/python

import os
import tensorflow as tf
import time

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CLASSES_COUNT = 10
TRAIN_FILE_NAME = 'cifar10-train.dat'
NUM_EPOCHS = 5
BATCH_SIZE = 100
MODEL_DIR = 'model'
MODEL_FILE_PREFIX = os.path.join(MODEL_DIR, 'cifarmodel')
METAGRAPH_FILE = 'cifarmetagraph'
SUMMARY_DIR = 'summaries'

def random_initializer():
  return tf.random_uniform_initializer(-0.1, 0.1)

def build_conv_layer(input_tensor, kernel_width, kernel_height, kernels_count):
  in_channels = int(input_tensor.shape[-1])
  kernels = tf.get_variable(
      'kernels',
      dtype=tf.float32,
      shape=[kernel_height, kernel_width, in_channels, kernels_count],
      initializer=random_initializer())
  conv_output = tf.nn.conv2d(
      input_tensor,
      kernels,
      strides=[1, 1, 1, 1],
      padding='SAME')
  conv_relu = tf.nn.relu(conv_output)
  pooled = tf.nn.pool(
      conv_relu,
      window_shape=[kernel_height, kernel_width],
      strides=[kernel_height, kernel_width],
      pooling_type='MAX',
      padding='SAME')
  return pooled


def flatten(input_tensor):
  dims = input_tensor.shape
  new_dim_size = 1
  for dim in dims[1:]:
    new_dim_size *= int(dim)
  return tf.reshape(input_tensor, [-1, new_dim_size])


def build_linear_layer(input_tensor, num_units, activation_function):
  input_sample_size = int(input_tensor.shape[-1])
  weights = tf.get_variable(
      'weights',
      dtype=tf.float32,
      shape=[input_sample_size, num_units],
      initializer=random_initializer())
  biases = tf.get_variable(
      'biases',
      dtype=tf.float32,
      shape=[1, num_units],
      initializer=random_initializer())
  linear_out = tf.matmul(input_tensor, weights) + biases
  if not activation_function:
    return linear_out
  return activation_function(linear_out, name='activations')


def build_accuracy(network_output, target_labels):
  predictions = tf.argmax(network_output, axis=1)
  target_predictions = tf.argmax(target_labels, axis=1)
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, target_predictions), tf.float32))
  return accuracy


def input_pipeline():
  filenames_queue = tf.train.string_input_producer(
      [TRAIN_FILE_NAME], num_epochs=NUM_EPOCHS)
  reader = tf.TFRecordReader()
  key, value = reader.read(filenames_queue)
  features = tf.parse_single_example(
      value,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)

  image = tf.reshape(image, (3, IMAGE_HEIGHT, IMAGE_WIDTH))
  image = tf.transpose(image, [1, 2, 0])
  image = tf.cast(image, tf.float32) * (1. / 256.)
  labels = tf.one_hot(
      label,
      depth=CLASSES_COUNT)
  images_batch, labels_batch = tf.train.shuffle_batch(
      [image, labels],
      batch_size=BATCH_SIZE,
      capacity=2000,
      min_after_dequeue=1000)
  return images_batch, labels_batch


def main():
  tf.set_random_seed(12345)
  input_image, target_labels = input_pipeline()

  with tf.variable_scope('conv1_layer'):
    current_output = build_conv_layer(input_image, 4, 4, 32)
  with tf.variable_scope('conv2_layer'):
    current_output = build_conv_layer(current_output, 4, 4, 32)
  with tf.variable_scope('reshape_layer'):
    current_output = flatten(current_output)
  with tf.variable_scope('full3_layer'):
    current_output = build_linear_layer(current_output, 75, tf.nn.relu)
  with tf.variable_scope('full4_layer'):
    current_output = build_linear_layer(current_output, 25, tf.nn.relu)
  with tf.variable_scope('output5_layer'):
    logits = build_linear_layer(current_output, CLASSES_COUNT, None)
    final_output = tf.nn.softmax(logits, name='activation')
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
       logits=logits,
       labels=target_labels))
  optimizer = tf.train.RMSPropOptimizer(0.0001, decay=0.9)
  train_step = optimizer.minimize(cross_entropy)

  accuracy = build_accuracy(final_output, target_labels)

  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.summary.scalar('accuracy', accuracy)
  merged_summaries = tf.summary.merge_all()
  model_saver = tf.train.Saver(max_to_keep=5)

  model_info = [input_image, target_labels, final_output]
  for param in model_info:
    tf.add_to_collection('model_info', param)
  model_saver.export_meta_graph(
      os.path.join(MODEL_DIR, METAGRAPH_FILE),
      collection_list=['model_info'])

  prev_step_end = time.time()
  with tf.Session() as session:
    with session.as_default():
      summary_writer = tf.summary.FileWriter(SUMMARY_DIR, session.graph)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      coordinator = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=session, coord=coordinator)
      try:
        step = 0
        while not coordinator.should_stop():
          step += 1
          session.run(train_step)
          if step % 10 == 0:
            ce, acc, summary = session.run(
                [cross_entropy, accuracy, merged_summaries])
            now = time.time()
            print(('After {} steps cross entropy {} accuracy {}' +
                  ' time since prev. report {} sec.').format(
                step, ce, acc, now-prev_step_end))
            prev_step_end = now
            summary_writer.add_summary(summary, step)
            model_saver.save(session, MODEL_FILE_PREFIX, step)
      except tf.errors.OutOfRangeError as ex:
        print('Training finished, {} steps done'.format(step))
      finally:
        file_path = model_saver.save(session, MODEL_FILE_PREFIX, step)
        print('Final model saved to {}'.format(file_path))
        summary_writer.close()
        coordinator.request_stop()

if __name__ == '__main__':
  main()

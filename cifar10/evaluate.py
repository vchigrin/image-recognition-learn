#!/usr/bin/env python

import sys
import tensorflow as tf

TRAIN_FILE_NAME = 'cifar10-train.dat'
TEST_FILE_NAME = 'cifar10-test.dat'
BATCH_SIZE = 100
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

def build_input_operation(session, file_name):
  reader = tf.TFRecordReader()
  filenames_queue = tf.train.string_input_producer([file_name], num_epochs=1)

  key, value = reader.read(filenames_queue)
  features = tf.parse_single_example(
      value,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int64)

  image = tf.reshape(image, (3, IMAGE_HEIGHT, IMAGE_WIDTH))
  image = tf.transpose(image, [1, 2, 0])
  image = tf.cast(image, tf.float32) * (1. / 256.)
  with session.as_default():
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
  return tf.train.batch(
      [image, label],
      batch_size=BATCH_SIZE)


def load_model(file_name, session):
  with session.as_default():
    saver = tf.train.import_meta_graph(file_name + '.meta')
    saver.restore(session, file_name)
    (input_image, target_labels, final_output) = tf.get_collection(
        'model_info')
  return session, input_image, final_output


def main():
  if len(sys.argv) != 3:
    print('Usage: {0} (--train|--test) model_file_path'.format(sys.argv[0]))
    return 1
  if (sys.argv[1] == '--train'):
    data_file_name = TRAIN_FILE_NAME
  else:
    data_file_name = TEST_FILE_NAME
  session = tf.Session()
  image_batch, label_batch = build_input_operation(session, data_file_name)
  session, input_var, result_prediction = load_model(sys.argv[2], session)
  predicted_labels = tf.argmax(result_prediction, axis=1)
  labels_var = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int64)
  num_matches = tf.reduce_sum(
      tf.cast(tf.equal(predicted_labels, labels_var), tf.int32))
  total_num_samples = 0
  total_num_matches = 0
  with session:
    with session.as_default():
      coordinator = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=session, coord=coordinator)
      try:
        while True:
          img, labels = session.run([image_batch, label_batch])
          num_matches_result = session.run(
              num_matches,
              feed_dict={input_var:img,
                         labels_var:labels})
          total_num_samples += BATCH_SIZE
          total_num_matches += num_matches_result
          sys.stdout.write('{} samples evaluated\r'.format(total_num_samples))
          sys.stdout.flush()
      except tf.errors.OutOfRangeError:
        print 'Result {}/{} ({} %)'.format(
            total_num_matches, total_num_samples,
            float(total_num_matches) / total_num_samples)
      finally:
        coordinator.request_stop()


if __name__ == '__main__':
  sys.exit(main())

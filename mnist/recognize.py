#!/usr/bin/env python
import copy
import numpy as np
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import itertools

N_L0_UNITS = 25
N_L1_UNITS = 15
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
LEARN_RATE = 0.05
N_GENERATIONS = 300
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.05
# If validation error will be worse then the best seen that number
# of epochs in row, we'll stop learning and use best model that we've found.
NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP = 10

class Network(object):
  def __init__(self):
    self._l0_coef = Network._rand_matrix(N_L0_UNITS, INPUT_SIZE)
    self._l0_bias = Network._rand_matrix(N_L0_UNITS, 1)
    self._l1_coef = Network._rand_matrix(N_L1_UNITS, N_L0_UNITS)
    self._l1_bias = Network._rand_matrix(N_L1_UNITS, 1)
    self._out_coef = Network._rand_matrix(N_OUTPUT_UNITS, N_L1_UNITS)
    self._out_bias = Network._rand_matrix(N_OUTPUT_UNITS, 1)

  @staticmethod
  def _rand_matrix(n_rows, n_columns):
    return np.random.rand(n_rows, n_columns) - 0.5

  @staticmethod
  def _sigmoid(val):
    return 1 / (1 + np.exp(-val))

  @staticmethod
  def _relu(val):
    result = val.copy()
    result[result < 0] *= 0.1
    return result

  @staticmethod
  def _softmax(input_values):
    values = input_values.copy()
    values = values - values.min()
    val_exp = np.exp(values)
    denominator = val_exp.sum()
    result = val_exp / denominator
    return result

  def learn_batch(self, sample_matrices, labels, learn_rate):
    out_coef_gradients = np.zeros(self._out_coef.shape)
    out_bias_gradients = np.zeros(self._out_bias.shape)
    l1_coef_gradients = np.zeros(self._l1_coef.shape)
    l1_bias_gradients = np.zeros(self._l1_bias.shape)
    l0_coef_gradients = np.zeros(self._l0_coef.shape)
    l0_bias_gradients = np.zeros(self._l0_bias.shape)
    batch_size = 0
    for sample_matrix, label in itertools.izip(sample_matrices, labels):
      out_coef, out_bias, l1_coef, l1_bias, l0_coef, l0_bias = self._process_sample(
          sample_matrix,  label)
      out_coef_gradients += out_coef
      out_bias_gradients += out_bias
      l1_coef_gradients += l1_coef
      l1_bias_gradients += l1_bias
      l0_coef_gradients += l0_coef
      l0_bias_gradients += l0_bias
      batch_size += 1
    # Average over all samples:
    out_coef_gradients /= batch_size
    out_bias_gradients /= batch_size
    l1_coef_gradients /= batch_size
    l1_bias_gradients /= batch_size
    l0_coef_gradients /= batch_size
    l0_bias_gradients /= batch_size
    assert not np.any(np.isnan(out_coef_gradients))
    assert not np.any(np.isnan(out_bias_gradients))
    assert not np.any(np.isnan(l1_coef_gradients))
    assert not np.any(np.isnan(l1_bias_gradients))
    assert not np.any(np.isnan(l0_coef_gradients))
    assert not np.any(np.isnan(l0_bias_gradients))
    # Update:
    self._out_coef -= learn_rate * out_coef_gradients
    self._out_bias -= learn_rate * out_bias_gradients
    self._l1_coef -= learn_rate * l1_coef_gradients
    self._l1_bias -= learn_rate * l1_bias_gradients
    self._l0_coef -= learn_rate * l0_coef_gradients
    self._l0_bias -= learn_rate * l0_bias_gradients


  def _process_sample(self, sample_data, label):
    assert not np.any(np.isnan(self._l0_coef))
    assert not np.any(np.isnan(self._l0_bias))
    assert not np.any(np.isnan(self._l1_coef))
    assert not np.any(np.isnan(self._l1_bias))
    assert len(sample_data) == INPUT_SIZE
    sample_data = sample_data.reshape(INPUT_SIZE, 1)
    l0_activations_input = np.dot(self._l0_coef, sample_data) + self._l0_bias
    assert l0_activations_input.shape == (N_L0_UNITS, 1)
    l0_activations = Network._relu(l0_activations_input)

    l1_activations_input = np.dot(self._l1_coef, l0_activations) + self._l1_bias
    assert l1_activations_input.shape == (N_L1_UNITS, 1)
    l1_activations = Network._relu(l1_activations_input)

    out_activations_input = np.dot(self._out_coef, l1_activations) + self._out_bias
    assert out_activations_input.shape == (N_OUTPUT_UNITS, 1)
    out_activations = Network._softmax(out_activations_input)

    expected_output = np.zeros([N_OUTPUT_UNITS, 1])
    expected_output[label, 0] = 1
    # Back-propagate errors
    common_out_grad = out_activations - expected_output
    assert common_out_grad.shape == (N_OUTPUT_UNITS, 1)
    out_coef_gradients = np.dot(common_out_grad, np.transpose(l1_activations))
    assert out_coef_gradients.shape == self._out_coef.shape
    out_bias_gradients = common_out_grad

    l1_grad_input = np.dot(np.transpose(self._out_coef), common_out_grad)
    assert l1_grad_input.shape == (N_L1_UNITS, 1)
    common_l1_grad = l1_activations.copy()
    common_l1_grad[common_l1_grad > 0] = 1
    common_l1_grad[common_l1_grad < 0] = 0.1
    common_l1_grad *= l1_grad_input

    assert common_l1_grad.shape == (N_L1_UNITS, 1)
    l1_coef_gradients = np.dot(common_l1_grad, np.transpose(l0_activations))
    assert l1_coef_gradients.shape == self._l1_coef.shape
    l1_bias_gradients = common_l1_grad

    l0_grad_input = np.dot(np.transpose(self._l1_coef), common_l1_grad)
    assert l0_grad_input.shape == (N_L0_UNITS, 1)
    common_l0_grad = l0_activations.copy()
    common_l0_grad[common_l0_grad > 0] = 1
    common_l0_grad[common_l0_grad < 0] = 0.1
    common_l0_grad *= l0_grad_input

    assert common_l0_grad.shape == (N_L0_UNITS, 1)
    l0_coef_gradients = np.dot(common_l0_grad, np.transpose(sample_data))
    assert l0_coef_gradients.shape == self._l0_coef.shape
    l0_bias_gradients = common_l0_grad
    return (out_coef_gradients, out_bias_gradients,
        l1_coef_gradients, l1_bias_gradients,
        l0_coef_gradients, l0_bias_gradients)

  def recognize_sample(self, sample_data):
    # INPUT_SIZE x 1
    sample_data = sample_data.reshape(INPUT_SIZE, 1)
    l0_activation_input = np.dot(self._l0_coef, sample_data) + self._l0_bias
    assert l0_activation_input.shape == (N_L0_UNITS, 1)
    l0_activations = Network._relu(l0_activation_input)

    l1_activations_input = np.dot(self._l1_coef, l0_activations) + self._l1_bias
    assert l1_activations_input.shape == (N_L1_UNITS, 1)
    l1_activations = Network._relu(l1_activations_input)

    out_activations_input = np.dot(self._out_coef, l1_activations) + self._out_bias
    assert out_activations_input.shape == (N_OUTPUT_UNITS, 1)
    out_activations = Network._softmax(out_activations_input)
    return np.argmax(out_activations)

def count_errors(network, stream):
  num_errors = 0
  num_examples = 0
  for sample_matrices, labels in stream.get_epoch_iterator():
    for sample, label in itertools.izip(sample_matrices, labels):
      num_examples += 1
      output_label = network.recognize_sample(sample)
      if label[0] != output_label:
        num_errors += 1
  return num_errors, num_examples

def count_errors_scheme(network, data, labels, iteration_scheme):
  num_errors = 0
  num_examples = 0
  for request in iteration_scheme.get_request_iterator():
    for index in request:
      num_examples += 1
      sample = data[index]
      label = labels[index]
      output_label = network.recognize_sample(sample)
      if label != output_label:
        num_errors += 1
  return num_errors, num_examples

def count_errors_lists(network, data_items, label_items):
  num_errors = 0
  for sample, label in itertools.izip(data_items, label_items):
    output_label = network.recognize_sample(sample)
    if label != output_label:
      num_errors += 1
  return num_errors


def main_fuel():
  dataset = fuel.datasets.mnist.MNIST(('train',))
  network = Network()
  num_train_examples = int(dataset.num_examples * (1 - VALIDATION_DATA_PART))
  train_stream = fuel.transformers.Flatten(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=fuel.schemes.SequentialScheme(
              examples=num_train_examples,
              batch_size=BATCH_SIZE)))
  validation_stream = fuel.transformers.Flatten(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=fuel.schemes.SequentialScheme(
              examples=range(num_train_examples,  dataset.num_examples),
              batch_size=BATCH_SIZE)))
  for i in xrange(N_GENERATIONS):
    print '----Generation {}'.format(i)
#    samples = dataset.get_data(h,None)
#    for data, label in itertools.izip(samples[0], samples[1]):
    for data, label in train_stream.get_epoch_iterator():
      network.learn_batch(data, label)
    num_errors, num_samples = count_errors(network, train_stream)
    print 'Training set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_samples, num_samples, num_errors)
    num_errors, num_samples = count_errors(network, validation_stream)
    print 'Validation set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_samples, num_samples, num_errors)
    train_stream.next_epoch()
    validation_stream.next_epoch()
  stream.close()

def to_batches(items):
  batch = []
  it = items.__iter__()
  while True:
    if len(batch) == BATCH_SIZE:
      yield batch
      batch = []
    else:
      try:
        batch.append(it.next())
      except StopIteration:
        break
  if batch:
    yield batch


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

def main():
  network = Network()
  data, labels = load_csv('kaggle/train.csv', True)
  num_validation_examples = int(len(data) * VALIDATION_DATA_PART)
  num_train_examples = len(data) - num_validation_examples
  print 'Data loaded. Train {} validation {}'.format(
       num_train_examples, num_validation_examples)
#  train_data = data[:num_train_examples]
#  train_labels = labels[:num_train_examples]
#  validation_data = data[num_train_examples:]
#  validation_labels = labels[num_train_examples:]
  best_net = None
  best_validation_errors = 0
  cross_validation_generator = fuel.schemes.cross_validation(
      fuel.schemes.SequentialScheme,
      num_examples=len(data),
      num_folds = int(1/VALIDATION_DATA_PART),
      strict=True,
      batch_size=BATCH_SIZE)
  cross_validation_schemes = list(cross_validation_generator)
  num_worse = 0
  for i in xrange(N_GENERATIONS):
    train_scheme, validation_scheme = cross_validation_schemes[
        i % len(cross_validation_schemes)]
    print '----Train Generation {}'.format(i)
    for request in train_scheme.get_request_iterator():
      samples = [data[p] for p in request]
      cur_labels = [labels[p] for p in request]
      network.learn_batch(samples, cur_labels)
    num_errors, num_examples = count_errors_scheme(network, data, labels, train_scheme)
    print 'Training set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_examples, num_examples, num_errors)
    num_errors, num_examples = count_errors_scheme(network, data, labels, validation_scheme)
    print 'Validation set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_examples, num_examples, num_errors)
    if best_net is None or num_errors <= best_validation_errors:
      print 'Updating best model'
      best_net = copy.deepcopy(network)
      best_validation_errors = num_errors
      num_worse = 0
    else:
      print 'We get WORSE results. on {} iteration. Total bad results {}'.format(i, num_worse)
      num_worse += 1
      if num_worse >= NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP:
        break
  print 'Training finished. Write result...'
  data, _ = load_csv('kaggle/test.csv', False)
  with open('kaggle/report-vchigrin.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index, sample in enumerate(data):
      label = best_net.recognize_sample(sample)
      f.write('{},{}\n'.format(index + 1, label))


if __name__ == '__main__':
  main()

#!/usr/bin/env python
import copy
import numpy as np
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import itertools

N_OUTPUT_UNITS = 10
N_L0_UNITS = 20
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
LEARN_RATE = 0.1
N_GENERATIONS = 300
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.05

class Network(object):
  def __init__(self):
    self._l0_coef = Network._rand_matrix(N_L0_UNITS, INPUT_SIZE)
    self._l0_bias = Network._rand_matrix(N_L0_UNITS, 1)
    self._l1_coef = Network._rand_matrix(N_OUTPUT_UNITS, N_L0_UNITS)
    self._l1_bias = Network._rand_matrix(N_OUTPUT_UNITS, 1)

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

  def learn_batch(self, sample_matrices, labels):
    l1_coef_gradients = np.zeros(self._l1_coef.shape)
    l1_bias_gradients = np.zeros(self._l1_bias.shape)
    l0_coef_gradients = np.zeros(self._l0_coef.shape)
    l0_bias_gradients = np.zeros(self._l0_bias.shape)
    batch_size = 0
    for sample_matrix, label in itertools.izip(sample_matrices, labels):
      l1_coef, l1_bias, l0_coef, l0_bias = self._process_sample(
          sample_matrix,  label)
      l1_coef_gradients += l1_coef
      l1_bias_gradients += l1_bias
      l0_coef_gradients += l0_coef
      l0_bias_gradients += l0_bias
      batch_size += 1
    # Average over all samples:
    l1_coef_gradients /= batch_size
    l1_bias_gradients /= batch_size
    l0_coef_gradients /= batch_size
    l0_bias_gradients /= batch_size
    assert not np.any(np.isnan(l1_coef_gradients))
    assert not np.any(np.isnan(l1_bias_gradients))
    assert not np.any(np.isnan(l0_coef_gradients))
    assert not np.any(np.isnan(l0_bias_gradients))
    # Update:
    self._l1_coef -= LEARN_RATE * l1_coef_gradients
    self._l1_bias -= LEARN_RATE * l1_bias_gradients
    self._l0_coef -= LEARN_RATE * l0_coef_gradients
    self._l0_bias -= LEARN_RATE * l0_bias_gradients


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
    assert l1_activations_input.shape == (N_OUTPUT_UNITS, 1)
    l1_activations = Network._softmax(l1_activations_input)

    expected_output = np.zeros([N_OUTPUT_UNITS, 1])
    expected_output[label, 0] = 1
    # Back-propagate errors
    common_l1_grad = l1_activations - expected_output
    assert common_l1_grad.shape == (N_OUTPUT_UNITS, 1)
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

    return (l1_coef_gradients, l1_bias_gradients,
        l0_coef_gradients, l0_bias_gradients)

  def recognize_sample(self, sample_data):
    # INPUT_SIZE x 1
    sample_data = sample_data.reshape(INPUT_SIZE, 1)
    l0_activation_input = np.dot(self._l0_coef, sample_data) + self._l0_bias
    assert l0_activation_input.shape == (N_L0_UNITS, 1)
    l0_activations = Network._relu(l0_activation_input)

    l1_activations_input = np.dot(self._l1_coef, l0_activations) + self._l1_bias
    assert l1_activations_input.shape == (N_OUTPUT_UNITS, 1)
    l1_activations = Network._softmax(l1_activations_input)
    return np.argmax(l1_activations)

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
  train_data = data[:num_train_examples]
  train_labels = labels[:num_train_examples]
  validation_data = data[num_train_examples:]
  validation_labels = labels[num_train_examples:]
  best_net = None
  best_validation_errors = 0
  for i in xrange(N_GENERATIONS):
    print '----Train Generation {}'.format(i)
    for pairs in to_batches(itertools.izip(train_data, train_labels)):
      samples = [p[0] for p in pairs]
      labels = [p[1] for p in pairs]
      network.learn_batch(samples, labels)
    num_errors = count_errors_lists(network, train_data, train_labels)
    print 'Training set error rate {} based on {} samples ({})'.format(
        float(num_errors) / len(train_data), len(train_data), num_errors)
    num_errors = count_errors_lists(network, validation_data, validation_labels)
    print 'Validation set error rate {} based on {} samples ({})'.format(
        float(num_errors) / len(validation_data), len(validation_data), num_errors)
    if best_net is None or num_errors <= best_validation_errors:
      print 'Updating best model'
      best_net = copy.deepcopy(network)
      best_validation_errors = num_errors
    else:
      print 'We get WORSE results. stopping on {} iteration'.format(i)
      break
  num_errors = count_errors_lists(best_net, validation_data, validation_labels)
  print 'Validation set BEST NET error rate {} based on {} samples ({})'.format(
      float(num_errors) / len(validation_data), len(validation_data), num_errors)
  print 'Training finished. Write result...'
  data, _ = load_csv('kaggle/test.csv', False)
  with open('kaggle/report-vchigrin.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index, sample in enumerate(data):
      label = best_net.recognize_sample(sample)
      f.write('{},{}\n'.format(index + 1, label))


if __name__ == '__main__':
  main()

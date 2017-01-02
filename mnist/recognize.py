#!/usr/bin/env python
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
N_GENERATIONS = 10
BATCH_SIZE = 100

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
  def _norm_sample(sample_matrix):
    assert sample_matrix.shape == (INPUT_WIDTH, INPUT_HEIGHT)
    return sample_matrix.reshape(INPUT_WIDTH * INPUT_WIDTH, 1)

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
    #print l0_activations_input
    assert l0_activations_input.shape == (N_L0_UNITS, 1)
    l0_activations = Network._sigmoid(l0_activations_input)

    l1_activations_input = np.dot(self._l1_coef, l0_activations) + self._l1_bias
    assert l1_activations_input.shape == (N_OUTPUT_UNITS, 1)  
    l1_activations = Network._sigmoid(l1_activations_input)

    expected_output = np.zeros([N_OUTPUT_UNITS, 1])
    expected_output[label, 0] = 1
    # Back-propagate errors
    common_l1_grad = (2 * (l1_activations - expected_output) * 
        l1_activations * (1 - l1_activations))
    assert common_l1_grad.shape == (N_OUTPUT_UNITS, 1)
    l1_coef_gradients = np.dot(common_l1_grad, np.transpose(l0_activations))
    assert l1_coef_gradients.shape == self._l1_coef.shape
    l1_bias_gradients = common_l1_grad

    l0_grad_input = np.dot(np.transpose(self._l1_coef), common_l1_grad)
    assert l0_grad_input.shape == (N_L0_UNITS, 1)
    common_l0_grad = l0_activations * (1 - l0_activations) * l0_grad_input
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
    l0_activations = Network._sigmoid(l0_activation_input)

    l1_activations_input = np.dot(self._l1_coef, l0_activations) + self._l1_bias
    assert l1_activations_input.shape == (N_OUTPUT_UNITS, 1)  
    l1_activations = Network._sigmoid(l1_activations_input)
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


def main():
  dataset = fuel.datasets.mnist.MNIST(('train',))
  network = Network()
  stream = fuel.transformers.Flatten(
      fuel.streams.DataStream.default_stream(
          dataset=dataset, 
          iteration_scheme=fuel.schemes.SequentialScheme(
              examples=dataset.num_examples, batch_size=BATCH_SIZE)))
  for i in xrange(N_GENERATIONS):
    print '----Generation {}'.format(i)
#    samples = dataset.get_data(h,None)
#    for data, label in itertools.izip(samples[0], samples[1]):
    for data, label in stream.get_epoch_iterator():
      network.learn_batch(data, label)
    num_errors, num_samples = count_errors(network, stream)
    print 'Training set error rate {} based on {} samples'.format(
        float(num_errors) / num_samples, num_samples)
    stream.next_epoch()
  stream.close()


if __name__ == '__main__':
  main()

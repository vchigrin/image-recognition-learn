#!/usr/bin/python -O

import collections
import copy
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sys
import time
import theano
import theano.d3viz
import theano.tensor as T
import theano.compile.nanguardmode

N_L0_UNITS = 85
N_L1_UNITS = 25
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
KERNEL_SIZE = 4
KERNELS_COUNT = 5
POOLING_SIZE = 4
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
START_LEARN_RATE = 0.01
MIN_LEARN_RATE = 0.0005
N_GENERATIONS = 200
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.1
# If validation error will be worse then the best seen that number
# of epochs in row, we'll stop learning and use best model that we've found.
NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP = 10
NUM_VALIDATION_SET_WORSINESS_TO_DECREASE_RATE = 2

DEFAULT_SEED = 12345

FUNCTION_MODE = None

if __debug__:
  FUNCTION_MODE = theano.compile.nanguardmode.NanGuardMode()

ValueStat = collections.namedtuple('ValueStat',
    ['min_val', 'max_val', 'avg_val'])

LayerStat = collections.namedtuple('LayerStat',
    ['weights_stat', 'biases_stat', 'weights_derivative', 'biases_derivative'])

class Layer(object):
  def forward_propagate(self, input_vector):
    raise NotImplemented()

  def num_params_matrices(self):
    raise NotImplemented()

  def compute_gradients_errors_vector(self, errors):
    """
    Returns one or more of vectors.
    |num_params_matrices| total - gradients for that
      layer's weights, biases, etc.

    Must be called immediatelly after forward_propagate (used values, cached
    by it).
    """
    raise NotImplemented()

  def update_params(self, params_update_vectors):
    """
    weights_update_vectors - vectors in the same order as produced
    by compute_gradients_errors_vector (shifted by one, that is without
    gradients relative to vector input.
    """
    raise NotImplemented()

  @staticmethod
  def _rand_matrix(n_rows, n_columns):
    return (np.random.rand(n_rows, n_columns) - 0.5) * 0.1


class WeightsBiasLayer(Layer):
  """
  Intermediate class for fully connected layers computing
  f(w*x+b)
  """
  def __init__(self, input_size, num_units, random_stream, layer_name):
    super(WeightsBiasLayer, self).__init__()
    self._input_size = input_size
    self._num_units = num_units
    self._weights = theano.shared(random_stream.uniform(
        low=-0.1, high=0.1,
        size=(num_units, input_size)),
        name='weights_' + layer_name)
    self._biases = theano.shared(random_stream.uniform(
        low=-0.1, high=0.1,
        size=(num_units, 1)),
        name='biases_' + layer_name)
    weights_update = T.dmatrix('weights_update')
    biases_update = T.dmatrix('biases_update')
    self._update_function = theano.function(
        [weights_update,  biases_update],
        [self._weights, self._biases],
        updates=[
          (self._weights, self._weights + weights_update),
          (self._biases, self._biases + biases_update),
        ])

  def _forward_propagate_with_function(self, input_vector, function):
    activations = T.dot(self._weights, input_vector)
    activations = activations + T.addbroadcast(self._biases, 1)
    return function(activations)

  def num_params_matrices(self):
    return 2

  def compute_gradients_errors_vector(self, errors):
    weight_gradients = T.grad(errors, self._weights,
        disconnected_inputs='raise', add_names=True)
    biases_gradients = T.grad(errors, self._biases,
        disconnected_inputs='raise', add_names=True);
    return [weight_gradients, biases_gradients]

  def get_weights(self):
    return self._weights.get_value()

  def get_biases(self):
    return self._biases.get_value()

  def update_params(self, params_update_vectors):
    weights_update, biases_update = params_update_vectors
    self._update_function(weights_update, biases_update)


def relu(val):
  return T.nnet.relu(val, 0.1)

def relu_grad_times_errors(relu_input, errors):
  return T.switch(T.lt(relu_input, 0), errors * 0.1, errors)

class ReLULayer(WeightsBiasLayer):
  """
  Fully connected to it input layer of ReLU units.
  """
  def forward_propagate(self, input_vector):
    return self._forward_propagate_with_function(input_vector, relu)


class SoftMaxLayer(WeightsBiasLayer):
  """
  Fully connected to it input layer of Softmax units.
  """
  def forward_propagate(self, input_vector):
    return self._forward_propagate_with_function(
        input_vector, SoftMaxLayer._softmax)

  @staticmethod
  def _softmax(input_values):
    # Softmax computes softmax along 1-st axis.
    return T.nnet.softmax(input_values.T).T


def compute_convolution_kernel_gradient_fast(
    input_matrix, output_matrix, kernel_shape):
  assert input_matrix.shape == output_matrix.shape
  # Note: [::-1,::-1] is equivalent to both np.flipud and np.fliplr.
  grad_src = scipy.signal.convolve2d(
    input_matrix[::-1,::-1], output_matrix, mode='full')[::-1,::-1]
  input_rows, input_columns = input_matrix.shape
  kernel_rows, kernel_columns = kernel_shape
  return grad_src[
      input_rows - 1 : input_rows - 1 + kernel_rows,
      input_columns - 1 : input_columns - 1 + kernel_columns]


class ConvolutionLayer(Layer):
  """
  Layer, that performs convoloution, ReLU to it output, and then performing
  max pooling.
  """
  def __init__(self, input_shape, kernel_size, kernels_count, pooling_size):
    super(ConvolutionLayer, self).__init__()
    self._kernels = []
    self._last_pooling_inputs = []
    for _ in xrange(kernels_count):
      self._kernels.append(Layer._rand_matrix(kernel_size, kernel_size))
    self._pooling_size = pooling_size
    self._input_shape = input_shape
    # Only support 2-D input at present
    assert len(input_shape) == 2
    # We do not expect pooling that drops any data or that uses filling.
    assert input_shape[0] % pooling_size == 0
    assert input_shape[1] % pooling_size == 0
    output_rows = input_shape[0] / pooling_size
    output_columns = input_shape[1] / pooling_size
    self._output_shape = (output_rows, output_columns)

  def output_shape(self):
    return (len(self._kernels), self._output_shape[0], self._output_shape[1])

  def forward_propagate(self, input_matrix):
    assert input_matrix.shape == self._input_shape
    self._last_convolution_input = input_matrix
    self._last_pooling_inputs = []
    for kernel in self._kernels:
      self._last_pooling_inputs.append(relu(scipy.signal.convolve2d(
          input_matrix, kernel, mode='same')))
    result = np.empty(self.output_shape(), dtype=np.float64)
    for index, pool_input in enumerate(self._last_pooling_inputs):
      result[index, :, :] = self._pool_layer(pool_input)
    return result

  def _pool_layer(self,  pooling_input):
    filtered = scipy.ndimage.filters.maximum_filter(
        pooling_input,  size=self._pooling_size)
    start = self._pooling_size / 2
    return filtered[start::self._pooling_size, start::self._pooling_size]

  def num_params_matrices(self):
    return len(self._kernels)

  def compute_gradients_errors_vector(self, errors):
    total_input_gradient = np.zeros(
        self._input_shape, dtype=np.float64)
    kernel_gradients = []
    assert len(self._kernels) == len(self._last_pooling_inputs)
    for layer_number, inputs in enumerate(itertools.izip(
          self._last_pooling_inputs, self._kernels)):
      (last_pooling_input, kernel) = inputs
      input_gradient, kernel_gradient = \
          self._compute_gradients_errors_vector_by_kernel(
              errors[layer_number,:,:],
              last_pooling_input, kernel)
      total_input_gradient += input_gradient
      kernel_gradients.append(kernel_gradient)
    result = [total_input_gradient]
    result.extend(kernel_gradients)
    return result

  def _compute_gradients_errors_vector_by_kernel(
      self, errors, last_pooling_input, kernel):
    assert errors.shape == self._output_shape
    maxpool_gradients = np.zeros(
        self._input_shape,
        dtype=np.float64)
    for error_row in xrange(self._output_shape[0]):
      for error_column in xrange(self._output_shape[1]):
        src_start_row = error_row * self._pooling_size
        src_start_column = error_column * self._pooling_size
        max_index = last_pooling_input[
            src_start_row : src_start_row + self._pooling_size,
            src_start_column :  src_start_column + self._pooling_size].argmax()
        src_row = src_start_row + max_index / self._pooling_size
        src_column = src_start_column + max_index % self._pooling_size
        maxpool_gradients[src_row, src_column] += errors[
            error_row, error_column]
    # Gradient needs only sign information. ReLU does not change sign,
    # so  using ReLU output instead of input seems safe.
    maxpool_gradients = relu_grad_times_errors(
        last_pooling_input, maxpool_gradients)
    # Convert gradients dE / d(convolution output) to
    # gradient dE / d(input data) and dE / d(kernel)
    input_gradients = scipy.signal.convolve2d(
        maxpool_gradients, kernel, mode='same')
    kernel_gradient = compute_convolution_kernel_gradient_fast(
        self._last_convolution_input, maxpool_gradients, kernel.shape)
    return [input_gradients, kernel_gradient]

  def update_params(self, params_update_vectors):
    assert len(self._kernels) == len(params_update_vectors)
    for index, update in enumerate(params_update_vectors):
      self._kernels[index] += update


class VectorizingLayer(Layer):
  """
  Very simple "Layer", do only reshape operation.
  Need only to simplify transition from convolution to
  fully connected layers.
  """
  def __init__(self, input_shape):
    super(VectorizingLayer, self).__init__()
    self._input_shape = input_shape
    total_elements = 1
    for dim in input_shape:
      total_elements *= dim
    self._output_shape = (total_elements, 1)

  def output_shape(self):
    return self._output_shape

  def forward_propagate(self, input_vector):
    assert input_vector.shape == self._input_shape
    return input_vector.reshape(self._output_shape)

  def num_params_matrices(self):
    return 0

  def compute_gradients_errors_vector(self, errors):
    assert errors.shape == self._output_shape
    return [errors.reshape(self._input_shape)]

  def update_params(self, params_update_vectors):
    pass


class Network(object):
  def __init__(self):
    self._layers = []
    random_stream = np.random.RandomState(seed=DEFAULT_SEED)
    self._layers.append(
        ReLULayer(INPUT_SIZE, N_L0_UNITS, random_stream, layer_name='layer0'))
    self._layers.append(
        ReLULayer(N_L0_UNITS, N_L1_UNITS, random_stream, layer_name='layer1'))
    self._layers.append(
        SoftMaxLayer(
            N_L1_UNITS, N_OUTPUT_UNITS, random_stream, layer_name='layer2'))
    self._layer_stat_lists = []
    for _ in xrange(len(self._layers)):
      self._layer_stat_lists.append([])

    input_theano_variable, output_theano_variable = \
        self._build_forward_propagate_function()
    self._build_backward_propagate_function(
        input_theano_variable, output_theano_variable)

  def _build_forward_propagate_function(self):
    input_data = T.dmatrix('input_data')
    cur_input = input_data
    for layer in self._layers:
      cur_input = layer.forward_propagate(cur_input)
    self._forward_propagate_function = theano.function(
        [input_data], cur_input,
        mode=FUNCTION_MODE)
    theano.d3viz.d3viz(
        cur_input,
        outfile="forward_propagate_unoptimized.html")
    theano.d3viz.d3viz(
        self._forward_propagate_function,
        outfile="forward_propagate_optimized.html")
    return input_data, cur_input

  def _build_backward_propagate_function(
      self, input_theano_variable, output_theano_variable):
    all_gradients = []

    expected_output = T.dmatrix('expected_output')
    # Back-propagate errors
    cost = T.nnet.categorical_crossentropy(
        output_theano_variable, expected_output).sum()
    cost.name = 'Cost'
    for layer in reversed(self._layers):
      gradients = layer.compute_gradients_errors_vector(cost)
      all_gradients.append(gradients)
    # Restore order of gradient update matrices
    result = []
    for gradients_list in reversed(all_gradients):
      result.extend(gradients_list)
    self._backward_propagate_function = theano.function(
         [input_theano_variable, expected_output], result,
        mode=FUNCTION_MODE)
    theano.d3viz.d3viz(
        result,
        outfile="backward_propagate_unoptimized.html")
    theano.d3viz.d3viz(
        self._backward_propagate_function,
        outfile="backward_propagate_optimized.html")

  @staticmethod
  def batch_matrices_to_tensor(sample_matrices):
    batch_size = len(sample_matrices)
    batch_matrices = np.empty((INPUT_SIZE, batch_size))
    for index, sample_matrix in enumerate(sample_matrices):
      batch_matrices[:, index] = sample_matrix
    return batch_matrices

  @staticmethod
  def batch_labels_to_tensor(labels):
    batch_size = len(labels)
    batch_output = np.zeros((N_OUTPUT_UNITS, batch_size))
    for index, label in enumerate(labels):
      batch_output[label, index] = 1
    return batch_output

  def learn_batch(self, sample_matrices, labels, learn_rate):
    batch_size = len(sample_matrices)
    batch_matrices = Network.batch_matrices_to_tensor(sample_matrices)
    batch_output = Network.batch_labels_to_tensor(labels)
    gradients = self._backward_propagate_function(
        batch_matrices, batch_output)
    # Update is equal to minus avg. gradient:
    updates = []
    grad_stats = []
    for grad in gradients:
      grad /= batch_size
      grad_stats.append(compute_stats(grad))
      updates.append(-grad)
    cur_index = 0
    weights_stats = []
    biases_stats = []
    for layer in self._layers:
      weights_stats.append(compute_stats(layer.get_weights()))
      biases_stats.append(compute_stats(layer.get_biases()))
      next_index = cur_index + layer.num_params_matrices()
      layer.update_params(updates[cur_index:next_index])
      cur_index = next_index
    assert len(grad_stats) == 2 * len(weights_stats)
    for index, lst in enumerate(self._layer_stat_lists):
      lst.append(LayerStat(
          weights_stats[index], biases_stats[index],
          grad_stats[index * 2], grad_stats[index * 2 + 1]))

  def get_layer_stats(self, layer_index):
     return self._layer_stat_lists[layer_index]

  def get_label_probabilities(self, sample_data):
    return self._forward_propagate_function(sample_data)

  def recognize_sample(self, sample_data):
    return np.argmax(self.get_label_probabilities(sample_data))


def compute_stats(tensor):
  minimum = tensor.min()
  maximum = tensor.max()
  avg = np.average(tensor)
  return ValueStat(minimum, maximum, avg)


def plot_values(stats_list, field_name):
  min_values = []
  max_values = []
  avg_values = []
  for layer_stat in stats_list:
    val_stat = layer_stat.__getattribute__(field_name)
    min_values.append(val_stat.min_val)
    max_values.append(val_stat.max_val)
    avg_values.append(val_stat.avg_val)
  plt.plot(min_values, 'b-', label='Minimum')
  plt.plot(max_values, 'r-', label='Maximum')
  plt.plot(avg_values, 'g-', label='Average')
  plt.xlabel('Batch')
  plt.ylabel('Stats')
  plt.legend()

def display_layer_stats(layer_index, stats_list):
  plt.suptitle('Layer {0} dynamics'.format(layer_index))

  plt.subplot(4, 1, 1)
  plt.title('Weights')
  plot_values(stats_list, 'weights_stat')

  plt.subplot(4, 1, 2)
  plt.title('Biases')
  plot_values(stats_list, 'biases_stat')

  plt.subplot(4, 1, 3)
  plt.title('Weights gradient')
  plot_values(stats_list, 'weights_derivative')

  plt.subplot(4, 1, 4)
  plt.title('Biases gradient')
  plot_values(stats_list, 'biases_derivative')

  plt.show()


def count_errors(network, stream):
  num_errors = 0
  num_examples = 0
  all_batches = list(stream.get_epoch_iterator())
  sum_cross_entropy = 0
  for index, batches in enumerate(all_batches):
    sys.stdout.write('Verify Batch {}/{}\r'.format(index, len(all_batches)))
    sys.stdout.flush()
    label_to_batch = dict(zip(stream.sources, batches))
    batch_matrices = Network.batch_matrices_to_tensor(
        label_to_batch['pixels'])
    expected_output = Network.batch_labels_to_tensor(
        label_to_batch['labels'])
    actual_output = network.get_label_probabilities(batch_matrices)
    assert actual_output.shape == expected_output.shape
    num_examples += len(label_to_batch['pixels'])
    samples_cross_entropy = np.log(actual_output) * expected_output
    sum_cross_entropy -= samples_cross_entropy.sum()
    predicted_labels = actual_output.argmax(axis=0)
    assert len(predicted_labels) == len(label_to_batch['labels'])
    for label, predicted_label in itertools.izip(
        label_to_batch['labels'], predicted_labels):
      if label != predicted_label:
        num_errors += 1
  return num_errors, num_examples, sum_cross_entropy


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


def make_image_matrix(input_batches):
  labels, input_arrays = input_batches
  new_arrays = []
  for array in input_arrays:
    new_arrays.append(array.reshape(INPUT_HEIGHT, INPUT_WIDTH))
  return (labels, new_arrays)

def main():
  if __debug__:
    theano.config.optimizer = 'None'
    theano.config.exception_verbosity = 'high'
  network = Network()
  dataset = fuel.datasets.H5PYDataset(
      'kaggle-mnist.hdf5', which_sets=('train',))
  print 'Data loaded. Total examples {}'.format(
       dataset.num_examples)
  best_net = None
  best_validation_errors = 0
  cross_validation_generator = fuel.schemes.cross_validation(
      fuel.schemes.SequentialScheme,
      num_examples=dataset.num_examples,
      num_folds = int(1/VALIDATION_DATA_PART),
      strict=True,
      batch_size=BATCH_SIZE)
  cross_validation_schemes = list(cross_validation_generator)
  num_worse = 0
  num_worse_for_rate = 0
  learn_rate = START_LEARN_RATE
  num_train_examples = int(dataset.num_examples * (1 - VALIDATION_DATA_PART))
  train_scheme = fuel.schemes.SequentialScheme(
      examples = num_train_examples,
      batch_size=BATCH_SIZE)
  validation_scheme = fuel.schemes.SequentialScheme(
      examples = range(num_train_examples, dataset.num_examples),
      batch_size=BATCH_SIZE)
  train_stream = fuel.transformers.Flatten(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=train_scheme))
  validation_stream = fuel.transformers.Flatten(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=validation_scheme))
  for i in xrange(N_GENERATIONS):
    print '----Train Generation {} at rate {}'.format(i, learn_rate)
    start_time = time.time()
    all_batches = list(train_stream.get_epoch_iterator())
    for index, batches in enumerate(all_batches):
      sys.stdout.write('Train Batch {}/{}\r'.format(index, len(all_batches)))
      sys.stdout.flush()
      label_to_batch = dict(zip(train_stream.sources, batches))
      network.learn_batch(
          label_to_batch['pixels'],
          label_to_batch['labels'], learn_rate)
    end_learn_time = time.time()
    num_errors, num_examples, sum_cross_entropy = count_errors(
        network, train_stream)
    print 'Training set cost {}, error rate {} based on {} samples ({})'.format(
        sum_cross_entropy / num_examples,
        float(num_errors) / num_examples, num_examples, num_errors)
    num_errors, num_examples, sum_cross_entropy = count_errors(
        network, validation_stream)
    end_validation_time = time.time()
    print 'Validation set cost {}, error rate {} based on {} samples ({})'.format(
        sum_cross_entropy / num_examples,
        float(num_errors) / num_examples, num_examples, num_errors)
    print(('Learning took {} sec.,' +
        ' validation data {} sec.,').format(
            end_learn_time - start_time,
            end_validation_time - end_learn_time))
    if best_net is None or sum_cross_entropy < best_sum_cross_entropy:
      print 'Updating best model'
      best_net = copy.deepcopy(network)
      best_sum_cross_entropy = sum_cross_entropy
      num_worse = 0
      num_worse_for_rate = 0
    else:
      num_worse += 1
      num_worse_for_rate += 1
      print 'We get WORSE results. on {} iteration. Total bad results {}'.format(i, num_worse)
      if num_worse >= NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP:
        break
      if num_worse_for_rate >= NUM_VALIDATION_SET_WORSINESS_TO_DECREASE_RATE:
        learn_rate = max(learn_rate / 2., MIN_LEARN_RATE)
        print 'DECREASING LEARN RATE TO {}'.format(learn_rate)
        num_worse_for_rate = 0

  for i_layer in xrange(3):
    display_layer_stats(i_layer, best_net.get_layer_stats(i_layer))
  print 'Training finished. Write result...'
  data, _ = load_csv('kaggle/test.csv', False)
  with open('kaggle/report-vchigrin.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index, sample in enumerate(data):
      sample = sample.reshape(INPUT_SIZE)
      label = best_net.recognize_sample(sample)
      f.write('{},{}\n'.format(index + 1, label))


if __name__ == '__main__':
  main()

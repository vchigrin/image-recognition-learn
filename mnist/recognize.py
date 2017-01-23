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
import theano.tensor.signal.pool
import theano.compile.nanguardmode

N_L0_UNITS = 75
N_L1_UNITS = 25
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
KERNEL_SIZE = 4
KERNELS_COUNT = 10
POOLING_SIZE = 4
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
START_LEARN_RATE = 0.001
WEIGHT_DECAY_RATE = 0.02
# How much use old, history date, relative to current batch gradients.
RMSPROP_DECAY_RATE = 0.9
MIN_LEARN_RATE = 0.0001
N_GENERATIONS = 200
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.1
# If validation error will be worse then the best seen that number
# of epochs in row, we'll stop learning and use best model that we've found.
NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP = 10
NUM_VALIDATION_SET_WORSINESS_TO_DECREASE_RATE = 2
COMPUTE_STATS = False

DEFAULT_SEED = 12345

FUNCTION_MODE = None

if __debug__:
  FUNCTION_MODE = theano.compile.nanguardmode.NanGuardMode()

ValueStat = collections.namedtuple('ValueStat',
    ['min_val', 'max_val', 'avg_val', 'sign_change_ratio'])

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
        size=(input_size, num_units)),
        name='weights_' + layer_name)
    self._biases = theano.shared(random_stream.uniform(
        low=-0.1, high=0.1,
        size=(1, num_units)),
        name='biases_' + layer_name)
    weights_update = T.dmatrix('weights_update')
    biases_update = T.dmatrix('biases_update')
    self._update_function = theano.function(
        [weights_update,  biases_update],
        [self._weights, self._biases],
        updates=[
          (self._weights, self._weights + weights_update),
          (self._biases, self._biases + biases_update),
        ],
        mode=FUNCTION_MODE)
    self._prev_weights = None
    self._prev_biases = None

  def get_regularization_expr(self):
    weights_squares_sum = (self._weights * self._weights).sum()
    return (WEIGHT_DECAY_RATE * weights_squares_sum) / 2

  def _forward_propagate_with_function(self, input_vector, function):
    activations = T.dot(input_vector, self._weights)
    activations = activations + T.addbroadcast(self._biases, 0)
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

  def get_prev_weights(self):
    return self._prev_weights

  def get_prev_biases(self):
    return self._prev_biases

  def update_params(self, params_update_vectors):
    self._prev_weights = self.get_weights()
    self._prev_biases = self.get_biases()
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
    return T.nnet.softmax(input_values)


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
  def __init__(self, input_shape, kernel_size, kernels_count, pooling_size,
      random_stream,  layer_name):
    super(ConvolutionLayer, self).__init__()
    self._kernels = theano.shared(
        random_stream.uniform(
            low=-0.1, high=0.1,
            size=(kernels_count, 1, kernel_size, kernel_size)),
        name=layer_name + '_kernels')
    self._kernels_count = kernels_count
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
    kernels_update = T.dtensor4('kernels_update')
    self._kernel_update_fun = theano.function(
       [kernels_update],
       [self._kernels],
       updates=[
         (self._kernels, self._kernels + kernels_update)
       ])

  def get_weights(self):
    return None

  def output_shape(self):
    return (self._kernels_count, self._output_shape[0], self._output_shape[1])

  def get_regularization_expr(self):
    return None

  def forward_propagate(self, input_matrix):
    conv_output = T.nnet.conv2d(
        input_matrix, self._kernels,
        border_mode='valid')
    pooled = T.signal.pool.pool_2d(
        conv_output,
        (self._pooling_size, self._pooling_size),
        mode='max')
    return T.nnet.relu(pooled)

  def num_params_matrices(self):
    return 1

  def compute_gradients_errors_vector(self, errors):
    kernel_grad = T.grad(errors, self._kernels,
        disconnected_inputs='raise', add_names=True)
    return [kernel_grad]

  def update_params(self, params_update_vectors):
    self._kernel_update_fun(params_update_vectors[0])


class VectorizingLayer(Layer):
  """
  Very simple "Layer", do only reshape operation.
  Need only to simplify transition from convolution to
  fully connected layers.
  """
  def __init__(self, input_shape):
    super(VectorizingLayer, self).__init__()
    # Describes shapre without 0-dimension (zero dimension - number of
    # training examples.
    self._input_shape = input_shape
    self._total_elements = 1
    for dim in input_shape:
      self._total_elements *= dim

  def total_elements(self):
    return self._total_elements

  def forward_propagate(self, input_tensor):
    batch_size = input_tensor.shape[0]
    result_size = (batch_size, self._total_elements)
    return input_tensor.reshape(result_size)

  def get_regularization_expr(self):
    return None

  def get_weights(self):
    return None

  def num_params_matrices(self):
    return 0

  def compute_gradients_errors_vector(self, errors):
    return []

  def update_params(self, params_update_vectors):
    pass


class Network(object):
  def __init__(self):
    self._layers = []
    random_stream = np.random.RandomState(seed=DEFAULT_SEED)

    self._layers.append(ConvolutionLayer(
        (INPUT_HEIGHT, INPUT_WIDTH),
        KERNEL_SIZE,
        KERNELS_COUNT,
        POOLING_SIZE,
        random_stream,
        layer_name='conv_layer0'))
    self._layers.append(VectorizingLayer(self._layers[-1].output_shape()))
   # self._layers.append(
   #     SoftMaxLayer(self._layers[-1].total_elements(), N_OUTPUT_UNITS, random_stream, layer_name='layer2'))
    self._layers.append(
        ReLULayer(self._layers[-1].total_elements(), N_L0_UNITS, random_stream, layer_name='layer0'))
    self._layers.append(
        ReLULayer(N_L0_UNITS, N_L1_UNITS, random_stream, layer_name='layer1'))
    self._layers.append(
        SoftMaxLayer(
            N_L1_UNITS, N_OUTPUT_UNITS, random_stream, layer_name='layer2'))
    self._layer_stat_lists = []
    for _ in xrange(len(self._layers) - 2):
      self._layer_stat_lists.append([])

    input_theano_variable, output_theano_variable = \
        self._build_forward_propagate_function()
    self._build_backward_propagate_function(
        input_theano_variable, output_theano_variable)
    self._accumulated_gradient_squares = None
    self._prev_gradients = None
    self._prev_params = None

  def _build_forward_propagate_function(self):
    input_data = T.dtensor4('input_data')
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
    for layer in self._layers:
      layer_regularization = layer.get_regularization_expr()
      if layer_regularization is not None:
        cost += layer_regularization
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
    batch_tensor = np.empty((batch_size, 1, INPUT_HEIGHT, INPUT_WIDTH))
    for index, sample_matrix in enumerate(sample_matrices):
      assert sample_matrix.shape == (INPUT_HEIGHT, INPUT_WIDTH)
      batch_tensor[index, 0, :, :] = sample_matrix
    return batch_tensor

  @staticmethod
  def batch_labels_to_tensor(labels):
    batch_size = len(labels)
    batch_output = np.zeros((batch_size, N_OUTPUT_UNITS))
    for index, label in enumerate(labels):
      batch_output[index, label] = 1
    return batch_output

  def learn_batch(self, sample_matrices, labels, learn_rate):
    batch_size = len(sample_matrices)
    batch_matrices = Network.batch_matrices_to_tensor(sample_matrices)
    batch_output = Network.batch_labels_to_tensor(labels)
    gradients = self._backward_propagate_function(
        batch_matrices, batch_output)
    # Update is equal to minus avg. gradient:
    updates = []
    normalized_gradients = [g / batch_size for g in gradients]
    if self._prev_gradients is not None:
      grad_stats = [compute_stats(g, old_g) for g, old_g in itertools.izip(
          normalized_gradients, self._prev_gradients)]
    else:
      grad_stats = [compute_stats(g, None) for g in normalized_gradients]
    self._prev_gradients = normalized_gradients
    if self._accumulated_gradient_squares is None:
      self._accumulated_gradient_squares = [
          g * g for g in normalized_gradients]
    else:
      assert len(normalized_gradients) == len(
          self._accumulated_gradient_squares)
      for index in xrange(len(normalized_gradients)):
        old_decayed = (self._accumulated_gradient_squares[index] *
            RMSPROP_DECAY_RATE)
        new = (normalized_gradients[index] * normalized_gradients[index] *
            (1 - RMSPROP_DECAY_RATE))
        self._accumulated_gradient_squares[index] = old_decayed + new

    for index in xrange(len(normalized_gradients)):
      # Add 10^-8 to prevent division-by-zero errors.
      norm_factor = -np.sqrt(
          10**(-8) + self._accumulated_gradient_squares[index])
      update = (normalized_gradients[index] * learn_rate) / norm_factor
      updates.append(update)
    cur_index = 0
    weights_stats = []
    biases_stats = []
    for layer in self._layers:
      weights = layer.get_weights()
      if weights is not None:
        weights_stats.append(compute_stats(
            weights, layer.get_prev_weights()))
        biases_stats.append(compute_stats(layer.get_biases(),
            layer.get_prev_biases()))
      next_index = cur_index + layer.num_params_matrices()
      layer.update_params(updates[cur_index:next_index])
      cur_index = next_index
    grad_offset = 1 # For 1-st convolution layer
    assert len(grad_stats) == 2 * len(weights_stats) + grad_offset
    for index, lst in enumerate(self._layer_stat_lists):
      lst.append(LayerStat(
          weights_stats[index], biases_stats[index],
          grad_stats[grad_offset + index * 2],
          grad_stats[grad_offset + index * 2 + 1]))

  def get_layer_stats(self, layer_index):
     return self._layer_stat_lists[layer_index]

  def get_label_probabilities(self, sample_data):
    return self._forward_propagate_function(sample_data)

  def recognize_sample(self, sample_data):
    return np.argmax(self.get_label_probabilities(sample_data))


def compute_stats(tensor, prev_tensor):
  if not COMPUTE_STATS:
    return None
  minimum = tensor.min()
  maximum = tensor.max()
  avg = np.average(tensor)
  if prev_tensor is None:
    sign_change_ratio = 0
  else:
    cur_sign = np.sign(tensor)
    prev_sign = np.sign(prev_tensor)
    sign_changes = (cur_sign != prev_sign).sum()
    sign_change_ratio = float(sign_changes) / cur_sign.size
  return ValueStat(minimum, maximum, avg, sign_change_ratio)


def plot_values(stats_list, field_name):
  min_values = []
  max_values = []
  avg_values = []
  sign_change_ratio_values = []
  for layer_stat in stats_list:
    val_stat = layer_stat.__getattribute__(field_name)
    min_values.append(val_stat.min_val)
    max_values.append(val_stat.max_val)
    avg_values.append(val_stat.avg_val)
    sign_change_ratio_values.append(val_stat.sign_change_ratio)
  plt.plot(min_values, 'b-', label='Minimum')
  plt.plot(max_values, 'r-', label='Maximum')
  plt.plot(avg_values, 'g-', label='Average')
  plt.plot(sign_change_ratio_values, 'y-', label='Sign change ratio')
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
    assert not np.any(np.isnan(actual_output))
    assert np.all(actual_output > 0)
    assert actual_output.shape == expected_output.shape
    num_examples += len(label_to_batch['pixels'])
    samples_cross_entropy = np.log(actual_output) * expected_output
    assert not np.any(np.isnan(samples_cross_entropy))
    sum_cross_entropy -= samples_cross_entropy.sum()
    predicted_labels = actual_output.argmax(axis=1)
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
  train_stream = fuel.transformers.Mapping(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=train_scheme),
      make_image_matrix)
  validation_stream = fuel.transformers.Mapping(
      fuel.streams.DataStream.default_stream(
          dataset=dataset,
          iteration_scheme=validation_scheme),
      make_image_matrix)
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

  if COMPUTE_STATS:
    for i_layer in xrange(3):
      display_layer_stats(i_layer, best_net.get_layer_stats(i_layer))
  print 'Training finished. Write result...'
  data, _ = load_csv('kaggle/test.csv', False)
  with open('kaggle/report-vchigrin.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index, sample in enumerate(data):
      sample = sample.reshape([INPUT_HEIGHT, INPUT_WIDTH])
      batch_matrix = Network.batch_matrices_to_tensor([sample])
      label = best_net.recognize_sample(batch_matrix)
      f.write('{},{}\n'.format(index + 1, label))


if __name__ == '__main__':
  main()

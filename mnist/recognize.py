#!/usr/bin/env python

import copy
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import itertools
import numpy as np
import scipy.signal
import sys
import time

N_L0_UNITS = 25
N_L1_UNITS = 15
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 28
INPUT_HEIGHT = 28
KERNEL_SIZE = 4
KERNELS_COUNT = 5
POOLING_SIZE = 4
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
START_LEARN_RATE = 0.1
MIN_LEARN_RATE = 0.0005
N_GENERATIONS = 200
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.1
# If validation error will be worse then the best seen that number
# of epochs in row, we'll stop learning and use best model that we've found.
NUM_VALIDATION_SET_WORSINESS_TO_GIVE_UP = 10
NUM_VALIDATION_SET_WORSINESS_TO_DECREASE_RATE = 4

class Layer(object):
  def forward_propagate(self, input_vector):
    raise NotImplemented()

  def num_params_matrices(self):
    raise NotImplemented()

  def compute_gradients_errors_vector(self, errors):
    """
    Returns one or more of vectors.
    First - error gradients relative to this layer input,
    required for error back propagation
    Second and further, |num_params_matrices| total - gradients for that
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
    return np.random.rand(n_rows, n_columns) - 0.5


class WeightsBiasLayer(Layer):
  """
  Intermediate class for fully connected layers computing
  f(w*x+b)
  """
  def __init__(self, input_size, num_units):
    super(WeightsBiasLayer, self).__init__()
    self._input_size = input_size
    self._num_units = num_units
    self._weights = Layer._rand_matrix(num_units, input_size)
    self._biases = Layer._rand_matrix(num_units, 1)

  def _forward_propagate_with_function(self, input_vector, function):
    assert not np.any(np.isnan(self._weights))
    assert not np.any(np.isnan(self._biases))
    assert input_vector.shape == (self._input_size, 1)
    self._last_input = input_vector
    self._last_activations_input = np.dot(self._weights, input_vector) + self._biases
    assert self._last_activations_input.shape == (self._num_units, 1)
    return function(self._last_activations_input)

  def num_params_matrices(self):
    return 2

  def _errorrs_in_function_grad(self, errors):
    """
    Converts errors gradients of f(w * x + b) to gradients for errors of
    (w * x + b). Subclasses must implement this.
    """
    raise NotImplemented()

  def compute_gradients_errors_vector(self, errors):
    assert errors.shape == (self._num_units, 1)
    assert not np.any(np.isnan(errors))
    errors_in_function_grad = self._errorrs_in_function_grad(errors)
    assert not np.any(np.isnan(errors_in_function_grad))
    assert errors_in_function_grad.shape == (self._num_units, 1)

    weight_gradients = np.dot(
        errors_in_function_grad, np.transpose(self._last_input))
    assert weight_gradients.shape == self._weights.shape

    input_gradients = np.dot(
      np.transpose(self._weights), errors_in_function_grad)
    assert input_gradients.shape == (self._input_size, 1)

    biases_gradients = errors_in_function_grad
    return [input_gradients, weight_gradients, biases_gradients]

  def update_params(self, params_update_vectors):
    weights_update, biases_update = params_update_vectors
    self._weights += weights_update
    self._biases += biases_update
    assert not np.any(np.isnan(self._weights))
    assert not np.any(np.isnan(self._biases))


def relu(val):
  result = val.copy()
  result[result < 0] *= 0.1
  return result

def relu_grad_times_errors(relu_input, errors):
  result = errors.copy()
  result[relu_input < 0] *= 0.1
  return result

class ReLULayer(WeightsBiasLayer):
  """
  Fully connected to it input layer of ReLU units.
  """
  def forward_propagate(self, input_vector):
    return self._forward_propagate_with_function(input_vector, relu)

  def _errorrs_in_function_grad(self, errors):
    return relu_grad_times_errors(self._last_activations_input, errors)

class SoftMaxLayer(WeightsBiasLayer):
  """
  Fully connected to it input layer of Softmax units.
  """
  def forward_propagate(self, input_vector):
    return self._forward_propagate_with_function(
        input_vector, SoftMaxLayer._softmax)

  def _errorrs_in_function_grad(self, errors):
    return errors

  @staticmethod
  def _softmax(input_values):
    val_exp = np.exp(input_values - input_values.min())
    denominator = val_exp.sum()
    result = val_exp / denominator
    return result


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
    self._layers.append(
        ConvolutionLayer(
            (INPUT_WIDTH, INPUT_HEIGHT),
            KERNEL_SIZE, KERNELS_COUNT, POOLING_SIZE))
    self._layers.append(VectorizingLayer(self._layers[-1].output_shape()))
    self._layers.append(ReLULayer(
        self._layers[-1].output_shape()[0], N_L0_UNITS))
    self._layers.append(ReLULayer(N_L0_UNITS, N_L1_UNITS))
    self._layers.append(SoftMaxLayer(N_L1_UNITS, N_OUTPUT_UNITS))

  def learn_batch(self, sample_matrices, labels, learn_rate):
    gradients = []
    batch_size = 0
    for sample_matrix, label in itertools.izip(sample_matrices, labels):
      sample_gradients = self._process_sample(sample_matrix, label)
      if not gradients:
        gradients = sample_gradients
      else:
        for index, grad in enumerate(sample_gradients):
          gradients[index] += grad
      batch_size += 1
    # Update is equal to minus avg. gradient:
    updates = []
    for grad in gradients:
      updates.append(-grad / batch_size)
    cur_index = 0
    for layer in self._layers:
      next_index = cur_index + layer.num_params_matrices()
      layer.update_params(updates[cur_index:next_index])
      cur_index = next_index

  def _process_sample(self, sample_data, label):
    assert sample_data.shape == (INPUT_HEIGHT, INPUT_WIDTH)
    cur_input = sample_data
    for layer in self._layers:
      cur_input = layer.forward_propagate(cur_input)
    expected_output = np.zeros([N_OUTPUT_UNITS, 1])
    expected_output[label, 0] = 1
    assert cur_input.shape == expected_output.shape
    # Back-propagate errors
    cur_errors = cur_input - expected_output
    all_gradients = []
    for layer in reversed(self._layers):
      gradients = layer.compute_gradients_errors_vector(cur_errors)
      cur_errors = gradients[0]
      all_gradients.append(gradients[1:])
    # Restore order of gradient update matrices
    result = []
    for gradients_list in reversed(all_gradients):
      result.extend(gradients_list)
    return result

  def recognize_sample(self, sample_data):
    assert sample_data.shape == (INPUT_HEIGHT, INPUT_WIDTH)
    cur_input = sample_data
    for layer in self._layers:
      cur_input = layer.forward_propagate(cur_input)
    return np.argmax(cur_input)


def count_errors(network, stream):
  num_errors = 0
  num_examples = 0
  for batches in stream.get_epoch_iterator():
    label_to_batch = dict(zip(stream.sources, batches))
    for sample, label in itertools.izip(
        label_to_batch['pixels'], label_to_batch['labels']):
      num_examples += 1
      output_label = network.recognize_sample(sample)
      if label[0] != output_label:
        num_errors += 1
  return num_errors, num_examples


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
      sys.stdout.write('Batch {}/{}\r'.format(index, len(all_batches)))
      sys.stdout.flush()
      label_to_batch = dict(zip(train_stream.sources, batches))
      network.learn_batch(
          label_to_batch['pixels'],
          label_to_batch['labels'], learn_rate)
    end_learn_time = time.time()
    num_errors, num_examples = count_errors(network, train_stream)
    print 'Training set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_examples, num_examples, num_errors)
    num_errors, num_examples = count_errors(network, validation_stream)
    end_validation_time = time.time()
    print 'Validation set error rate {} based on {} samples ({})'.format(
        float(num_errors) / num_examples, num_examples, num_errors)
    print(('Learning took {} sec.,' +
        ' validation data {} sec.,').format(
            end_learn_time - start_time,
            end_validation_time - end_learn_time))
    if best_net is None or num_errors < best_validation_errors:
      print 'Updating best model'
      best_net = copy.deepcopy(network)
      best_validation_errors = num_errors
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

  print 'Training finished. Write result...'
  data, _ = load_csv('kaggle/test.csv', False)
  with open('kaggle/report-vchigrin.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for index, sample in enumerate(data):
      sample = sample.reshape(INPUT_HEIGHT, INPUT_WIDTH)
      label = best_net.recognize_sample(sample)
      f.write('{},{}\n'.format(index + 1, label))


if __name__ == '__main__':
  main()

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
import theano
import theano.d3viz
import theano.tensor as T
import theano.tensor.signal.pool
import theano.compile.nanguardmode
import blocks.algorithms
import blocks.bricks
import blocks.bricks.conv
import blocks.bricks.cost
import blocks.extensions
import blocks.extensions.monitoring
import blocks.extensions.saveload
import blocks.main_loop
import blocks_extras.extensions.plot

N_L0_UNITS = 75
N_L1_UNITS = 25
N_OUTPUT_UNITS = 10
INPUT_WIDTH = 32
INPUT_HEIGHT = 32
KERNEL_SIZE = 4
KERNELS_COUNT = 32
POOLING_SIZE = 4
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT
LEARN_RATE = 0.001
WEIGHT_DECAY_RATE = 0#0.004
# How much use old, history date, relative to current batch gradients.
RMSPROP_DECAY_RATE = 0.9
N_EPOCHS = 30
BATCH_SIZE = 100
VALIDATION_DATA_PART = 0.1

FUNCTION_MODE = None

if __debug__:
  FUNCTION_MODE = theano.compile.nanguardmode.NanGuardMode()

class LeakyRectifier(blocks.bricks.Activation):
    def __init__(self, leak=0.01, **kwargs):
        super(LeakyRectifier, self).__init__(**kwargs)
        self._leak = leak

    @blocks.bricks.application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.nnet.relu(input_, alpha=self._leak)


def build_streams_for_dataset(dataset):
  num_train_examples = int(dataset.num_examples * (1 - VALIDATION_DATA_PART))
  validation_examples_range = range(num_train_examples, dataset.num_examples)
  train_scheme = fuel.schemes.SequentialScheme(
      examples=num_train_examples,
      batch_size=BATCH_SIZE)
  validation_scheme = fuel.schemes.SequentialScheme(
      examples=validation_examples_range,
      batch_size=BATCH_SIZE)
  train_stream = fuel.streams.DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=train_scheme)
  validation_stream = fuel.streams.DataStream.default_stream(
        dataset=dataset,
        iteration_scheme=validation_scheme)
  return  train_stream, validation_stream


def build_network(in_sample):
  conv_layer = blocks.bricks.conv.Convolutional(
      filter_size=(KERNEL_SIZE, KERNEL_SIZE),
      num_filters=KERNELS_COUNT,
      num_channels=3,
      image_size=(INPUT_WIDTH, INPUT_HEIGHT),
      border_mode='full')
  conv_layer.weights_init = blocks.initialization.Uniform(mean=0, width=0.1)
  conv_layer.biases_init = blocks.initialization.Constant(0)
  conv_output = LeakyRectifier().apply(conv_layer.apply(in_sample))

  pool_layer = blocks.bricks.conv.MaxPooling(
      input_dim=conv_layer.get_dim('output'),
      pooling_size=(POOLING_SIZE, POOLING_SIZE))

  pool_activations = pool_layer.apply(conv_output)

  flattener = blocks.bricks.conv.Flattener()
  flattener_output = flattener.apply(
      pool_activations)

  flattener_output_size = 1
  for dim in pool_layer.get_dim('output'):
      flattener_output_size *= dim
  layer0 = blocks.bricks.Linear(
      name='layer 0',
      input_dim=flattener_output_size,
      output_dim=N_L0_UNITS)

  layer0.weights_init = blocks.initialization.Uniform(mean=0, width=0.1)
  layer0.biases_init = blocks.initialization.Uniform(mean=0, width=0.1)
  layer0_activations = LeakyRectifier().apply(
      layer0.apply(flattener_output))

  layer1 = blocks.bricks.Linear(
      name='layer 1',
      input_dim=N_L0_UNITS,
      output_dim=N_L1_UNITS)
  layer1.weights_init = blocks.initialization.Uniform(mean=0, width=0.1)
  layer1.biases_init = blocks.initialization.Uniform(mean=0, width=0.1)
  layer1_activations = LeakyRectifier().apply(
      layer1.apply(layer0_activations))

  output_layer = blocks.bricks.Linear(
      name='output layer',
      input_dim=N_L1_UNITS,
      output_dim=N_OUTPUT_UNITS)
  output_layer.weights_init = blocks.initialization.Uniform(mean=0, width=0.1)
  output_layer.biases_init = blocks.initialization.Uniform(mean=0, width=0.1)
  output_activations = blocks.bricks.Softmax(
      name='computed_probabilities').apply(
          output_layer.apply(layer1_activations))
  conv_layer.initialize()
  pool_layer.initialize()
  layer0.initialize()
  layer1.initialize()
  output_layer.initialize()
  return output_activations


def main():
  if __debug__:
    theano.config.optimizer = 'None'
    theano.config.exception_verbosity = 'high'

  in_sample = T.tensor4('features')
  target = T.imatrix('targets')
  output_activations = build_network(in_sample)

  cost = blocks.bricks.cost.CategoricalCrossEntropy().apply(
      target.flatten(), output_activations)
  error_rate = blocks.bricks.cost.MisclassificationRate().apply(
      target.flatten(), output_activations)
  error_rate.name = 'error_rate'
  raw_cost = cost
  raw_cost.name = 'raw_cost'
  computation_graph = blocks.graph.ComputationGraph(cost)
  weights = blocks.filter.VariableFilter(roles=[blocks.roles.WEIGHT])(
      computation_graph.variables)
  for weight in weights:
    # Do not regularize Convolution layer kernels
    if blocks.roles.has_roles(weight, [blocks.roles.FILTER]):
      continue
    cost = cost + WEIGHT_DECAY_RATE * (weight ** 2).sum()
  cost.name = 'cost_with_regularization'
  algorithm = blocks.algorithms.GradientDescent(
      cost=cost,
      parameters=computation_graph.parameters,
      step_rule=blocks.algorithms.RMSProp(
          learning_rate=LEARN_RATE, decay_rate=RMSPROP_DECAY_RATE))

  dataset = fuel.datasets.CIFAR10(which_sets=('train',))
  print 'Data loaded. Total examples {}'.format(
       dataset.num_examples)
  train_stream, validation_stream = build_streams_for_dataset(
      dataset)

  monitor = blocks.extensions.monitoring.DataStreamMonitoring(
      variables=[raw_cost, cost, error_rate],
      data_stream=validation_stream,
      prefix="validation")
  train_monitor = blocks.extensions.monitoring.TrainingDataMonitoring(
      variables=[
          raw_cost, cost, error_rate,
          blocks.monitoring.aggregation.mean(algorithm.total_gradient_norm),
          blocks.monitoring.aggregation.mean(algorithm.total_step_norm),
          ],
      prefix="train",
      after_epoch=False,
      after_batch=True)
  main_loop = blocks.main_loop.MainLoop(
      data_stream=train_stream,
      algorithm=algorithm,
      model=computation_graph,
      extensions=[
          monitor,
          train_monitor,
          blocks.extensions.FinishAfter(after_n_epochs=N_EPOCHS),
          blocks.extensions.Timing(after_epoch=True),
          blocks.extensions.saveload.Checkpoint(
              'nn.dat', after_epoch=True),
          blocks_extras.extensions.plot.Plot(
              'Plotting CIFAR10 learning',
              channels=[
                ['train_raw_cost', 'validation_raw_cost',
                 'train_cost_with_regularization',
                 'validation_cost_with_regularization',],
                [ 'train_total_gradient_norm',
                 'train_total_step_norm',
                 ],
                ['train_error_rate', 'validation_error_rate'] ],
              after_batch=True),
          blocks.extensions.Printing(),
      ])
  main_loop.run()


if __name__ == '__main__':
  main()

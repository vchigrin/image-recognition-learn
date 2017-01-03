#!/usr/bin/env python
import copy
import numpy as np
import fuel.datasets
import fuel.schemes
import fuel.streams
import fuel.transformers
import h5py
import itertools

INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT


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
      assert len(data) == INPUT_SIZE
      data_arrays.append(data)
  return data_arrays, label_arrays

def main():
  print 'Loading data...'
  train_data, train_labels = load_csv('kaggle/train.csv', True)
  print 'Writing file...'
  f = h5py.File('kaggle-mnist.hdf5', mode='w')
  pixel_features = f.create_dataset(
      'pixels', (len(train_data), INPUT_SIZE), dtype='float32')
  labels = f.create_dataset(
      'labels', (len(train_data), 1), dtype='uint8')
  for index in xrange(len(train_data)):
    pixel_features[index] = train_data[index]
    labels[index] = train_labels[index]
  split_array = np.empty(
      2,
      dtype=np.dtype([
          ('split', 'a', 5),
          ('source', 'a', 15),
          ('start', np.int64, 1),
          ('stop', np.int64, 1),
          ('indices', h5py.special_dtype(ref=h5py.Reference)),
          ('available', np.bool, 1),
          ('comment', 'a', 1)]))
  split_array[:]['split'] = 'train'.encode('utf-8')
  split_array[0]['source'] = 'pixels'.encode('utf-8')
  split_array[1]['source'] = 'labels'.encode('utf-8')
  split_array[:]['start'] = 0
  split_array[:]['stop'] = len(train_data)
  split_array[:]['indices'] = h5py.Reference()
  split_array[:]['available'] = True
  split_array[:]['comment'] = '.'.encode('utf8')
  f.attrs['split'] = split_array
  f.flush()
  f.close()



if __name__ == '__main__':
  main()

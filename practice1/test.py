#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random
import time

m = 100
n = 1000
K = 200
lr = 0.1
use_vectorization = True

input_size = 2
output_size = 1

def random_generate_data(n: int):
  xs = np.random.uniform(-10, 10, (n, input_size))
  ys = (np.sum(xs, axis=1) > 0).astype(np.int32).reshape((-1,1))
  return xs, ys

# Randomly generate training, test data
x_train, y_train = random_generate_data(m)
x_test, y_test = random_generate_data(n)

# Initialize parameters
W = np.random.uniform(-1, 1, (input_size, output_size))
b = np.zeros(output_size)

def non_vectorized_matmul(a, b):
  res = []
  for i in range(a.shape[0]):
    s = 0.0
    for j in range(a.shape[1]):
      s += a[i][j] * b[j][0]
    res.append(s)
  return np.array(res).reshape((-1,1))

def vectorized_matmul(a, b):
  return np.matmul(a, b)

def sigmoid(x):
  return 0.98 / (1.0 + np.exp(-x)) + 0.01

def g_sigmoid(x):
  x_ = sigmoid(x)
  return x_ * (1 - x_) / 0.98

def cross_entropy(y, y_):
  return (y-1) * np.log(1 - y_) - y * np.log(y_)

def g_cross_entropy(y, y_):
  return (1-y)/(1-y_) - y/y_

def forward(x):
  if use_vectorization:
    s1 = vectorized_matmul(x, W) + b
  else:
    s1 = non_vectorized_matmul(x, W) + b
  return s1, sigmoid(s1)

def backward(x, y):
  global W, b
  s1, y_ = forward(x)
  g1 = g_cross_entropy(y, y_)
  g2 = g1 * g_sigmoid(s1)
  g3 = np.matmul(x.T, g2)
  W -= lr * g3
  b -= np.mean(lr * g2)

def test():
  s1, y_ = forward(x_test)
  L = cross_entropy(y_test, y_)
  test_acc = np.mean((y_test > 0.5) == (y_ > 0.5)) * 100
  train_acc = np.mean((y_train > 0.5) == (forward(x_train)[1] > 0.5)) * 100
  print('L:', np.mean(L), 'test_acc:', test_acc, 'train_acc:', train_acc, 'W:', W.reshape(-1), 'b:', b)

def save_pic(path):
  _, y_ = forward(x_test)
  indices = (y_ > 0.5).reshape(-1)
  plt.plot(x_test[indices,0], x_test[indices,1], 'bo')
  indices = (y_ <= 0.5).reshape(-1)
  plt.plot(x_test[indices,0], x_test[indices,1], 'ro')
  plt.savefig(path, dpi=300)

start_time = time.time()
save_pic('before.png')
for i in range(K//10):
  test()
  for _ in range(10):
    backward(x_train, y_train)
test()
end_time = time.time()
diff_time = end_time - start_time
print('time:', diff_time)
save_pic('after.png')

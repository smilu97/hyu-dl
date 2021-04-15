#!/usr/bin/env python

'''
title: HYU DeepLearning Practice #2
author: 2016025241 김영진
'''

import random
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='HYU DL Practice #2')
parser.add_argument('--M', type=int, help='the number of train data', default=1000)
parser.add_argument('--N', type=int, help='the number of test data', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--lr_decay', type=float, help='decay rate of learning rate', default=0.999)
parser.add_argument('--h1', type=int, help='the size of layer H1', default=3)
parser.add_argument('--h2', type=int, help='the size of layer H2', default=1)
parser.add_argument('--epoch', type=int, default=1000)

eps = 1e-9
input_size = 2
output_size = 1

def get_py_data(sz):
  x1_train, x2_train, y_train = [], [], []
  for i in range(sz):
    x1_train.append(random.uniform(-10, 10))
    x2_train.append(random.uniform(-10, 10))
    y_train.append(1 if x1_train[-1] + x2_train[-1] > 0 else 0)
  return x1_train, x2_train, y_train

def get_data(sz):
  x1_train, x2_train, y_train = get_py_data(sz)
  x = np.empty((sz, 2))
  y = np.empty((sz, 1))
  x[:, 0] = x1_train
  x[:, 1] = x2_train
  y[:, 0] = y_train
  return x, y

class Model:
  def __init__(self, h1=3, h2=1, lr=0.01, lr_decay=1.0):
    self.h1 = h1
    self.h2 = h2
    self.lr = lr
    self.lr_decay = lr_decay

    self.W1 = np.random.normal(size=(input_size, h1))
    self.W2 = np.random.normal(size=(h1, h2))
    self.W3 = np.random.normal(size=(h2, output_size))

    self.b1 = np.random.normal(size=(h1,))
    self.b2 = np.random.normal(size=(h2,))
    self.b3 = np.random.normal(size=(output_size,))
  
  @staticmethod
  def activate(x):
    return np.tanh(x)
  
  @staticmethod
  def d_activate(x):
    return (1-x)*(1+x)
  
  @staticmethod
  def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
  
  @staticmethod
  def d_sigmoid(x):
    return x*(1-x)
  
  @staticmethod
  def cross_entropy(y_, y):
    return -np.mean(y * np.log(y_ + eps) + (1-y) * np.log(1-y_ + eps))
  
  @staticmethod
  def d_cross_entropy(y_, y):
    return (-y/y_ + (1-y) / (1-y_)) / y_.shape[0]
  
  @staticmethod
  def mse(y_, y):
    return np.mean(np.square(y_ - y))
  
  @staticmethod
  def d_mse(y_, y):
    return (y_ - y) / y_.shape[0]
  
  def forward(self, x):
    self.x = x
    self.A1 = self.activate(np.matmul(x, self.W1) + self.b1)
    self.A2 = self.activate(np.matmul(self.A1, self.W2) + self.b2)
    self.y_ = self.sigmoid(np.matmul(self.A2, self.W3) + self.b3)
    return self.y_
  
  def loss(self, x, y):
    y_ = self.forward(x)
    return self.cross_entropy(y_, y)
  
  def fit(self, x, y):
    # AB * BC = AC
    # gA = C * B.T
    # gB = A.T * C
    y_ = self.forward(x)

    g = self.d_cross_entropy(y_, y)
    g *= self.d_sigmoid(y_)
    self.g_W3 = np.matmul(self.A2.T, g)
    self.g_b3 = np.sum(g, axis=0)

    g = np.matmul(g, self.W3.T)
    g *= self.d_activate(self.A2)
    self.g_W2 = np.matmul(self.A1.T, g)
    self.g_b2 = np.sum(g, axis=0)

    g = np.matmul(g, self.W2.T)
    g *= self.d_activate(self.A1)
    self.g_W1 = np.matmul(x.T, g)
    self.g_b1 = np.sum(g, axis=0)

    self.apply_gradient(self.W1, self.g_W1)
    self.apply_gradient(self.W2, self.g_W2)
    self.apply_gradient(self.W3, self.g_W3)
    self.apply_gradient(self.b1, self.g_b1)
    self.apply_gradient(self.b2, self.g_b2)
    self.apply_gradient(self.b3, self.g_b3)
    
    self.decay()
  
  def acc(self, x, y):
    y_ = self.forward(x)
    loss = self.cross_entropy(y_, y)
    pred = y_ > 0.5
    return np.mean(pred == y), loss
  
  def apply_gradient(self, v, g):
    v -= self.lr * g
  
  def decay(self):
    self.lr *= self.lr_decay

def main():
  args = parser.parse_args()

  M = args.M
  N = args.N
  lr = args.lr
  lr_decay = args.lr_decay
  h1 = args.h1
  h2 = args.h2
  epoch = args.epoch

  x_train, y_train = get_data(M)
  x_test,  y_test  = get_data(N)
  model = Model(h1=h1, h2=h2, lr=lr, lr_decay=lr_decay)
  
  for i in range(epoch):
    model.fit(x_train, y_train)
    if i % 10 == 9:
      acc, loss = model.acc(x_test, y_test)
      print('acc:', acc, 'loss:', loss)

if __name__ == '__main__':
  main()

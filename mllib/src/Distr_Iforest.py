#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Distr_Iforest.py
分布式孤立森林
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

from random import random, choice
from math import log

class Node(object):
    def __init__(self, size):
        self.size = size

        self.left = None
        self.right = None
        self.feature = None
        self.split = None


class IsolationTree(object):
    def __init__(self, X, n_samples, max_depth):
        self.depth = 1
        n = X.count()
        if n_samples > n:
            n_samples = n
        self.root = Node(n_samples)
        self._build_tree(X, n_samples, max_depth)

    def _get_split(self, idxs, feature):
        unique = idxs.select(feature).distinct()
        if unique.count() == 1:
            return None
        tem = unique.summary()
        x_min = float(tem.filter(tem['summary'] == 'min').select(feature).collect()[0][feature])
        x_max = float(tem.filter(tem['summary'] == 'max').select(feature).collect()[0][feature])
        #random产生[0,1)
        return random() * (x_max - x_min) + x_min

    def _build_tree(self, X, n_samples, max_depth):
        m = X.columns
        n = X.count()
        fra = round(float(n_samples)/n,4)
        idxs = X.sample(False, fra, 666) 
        que = [(self.depth + 1, self.root, idxs)]
        while que:
            depth, nd, idxs = que.pop(0)
            if depth > max_depth:
                depth -= 1
                break
            feature = choice(m)
            split = self._get_split(idxs, feature)
            if split is None:
                continue
            idxs_split = list_split(idxs, feature, split)
            nd.feature = feature
            nd.split = split
            nd.left = Node(idxs_split[0].count())
            nd.right = Node(idxs_split[1].count())
            que.append((depth + 1, nd.left, idxs_split[0]))
            que.append((depth + 1, nd.right, idxs_split[1]))
        self.depth = depth

    def _predict(self, Xi):
        nd = self.root
        depth = 0
        while nd.left and nd.right:
            if Xi[nd.feature] <= nd.split:
                nd = nd.left
            else:
                nd = nd.right
            depth += 1
        return depth, nd.size

def list_split(idxs, feature, split):
    ret = idxs.filter(idxs[feature] <= split), idxs.filter(idxs[feature] > split)

    return ret


class IsolationForest(object):
    def __init__(self):
        self.trees = None
        self.adjustment = None  

    def fit(self, X, n_samples=256, max_depth=10, n_trees=100):
        self.adjustment = self._get_adjustment(n_samples)
        self.trees = [IsolationTree(X, n_samples, max_depth)
                      for _ in range(n_trees)]

    def _get_adjustment(self, node_size):
        if node_size > 2:
            i = node_size - 1
            ret = 2 * (log(i) + 0.5772156649) - 2 * i / node_size
        elif node_size == 2:
            ret = 1
        else:
            ret = 0
        return ret

    def _predict(self, Xi):
        score = 0
        n_trees = len(self.trees)
        for tree in self.trees:
            depth, node_size = tree._predict(Xi)
            score += (depth + self._get_adjustment(node_size))
        score = score / n_trees

        return 2 ** -(score / self.adjustment)

    def predict(self, X):

        return X.rdd.map(self._predict)
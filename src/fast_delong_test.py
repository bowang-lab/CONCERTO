#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import norm
import os

"""
Created on Tue Nov  6 10:06:52 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""

import numpy as np
import scipy.stats
from scipy import stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def calculate_midranks(seq):
    """
    Calculates the mid-ranks of a sequence. So if sequence Z_1, ..., Z_M is input,
    outputs associated mid-ranks T_z(Z_1), ..., T_z(Z_M).
    We have that ``M`` is the length of the input sequence, ``I`` is a list of indices, where
    I[j] = i implies that the jth entry of ``seq`` should have an index i were ``seq`` to be sorted
    in ascending order. ``W`` is simply just ``seq`` but in ascending order. T[k] = c implies that
    the kth element of ``W`` has a midrank of c, and T_z[k] = c implies that the kth element of
    ``seq`` has a midrank of c when ``seq`` is sorted into ``W``.
    The indices ''i'' and ''j'' iterate through the list to find the beginning and end of subsequences of
    identical numbers within ''W''. If a subsequence is of length 1 (a unique number within the set of ''W''), then
    ''a'' = ''b'' = index of the unique number. If a subsequence is of length greater than 1, then
    ''a'' = index of beginning of subsequence in ''W'' < ''b'' = index of end of subsequence in ''W'', and the midrank
    of this subsequence is equal to (a+b)/2 by equation 11 in the paper cited above. The indices ''i'', ''j'', ''a'',
    and ''b'' are all 1-indexed, so when accessing memory in lists, we must subtract 1 from the index (i.e. W[a-1]).
    Parameters
    ----------
    seq: (Mx1) ndarray

    Returns
    -------
    ndarray
        of size (Mx1)
    """

    M = len(seq)
    I = np.argsort(seq, kind='quicksort')
    W = [seq[i] for i in I] + [np.max(seq) + 1]
    T = np.zeros(M)
    T_z = np.zeros(M)

    i = 1
    while i <= M:
        a = i
        j = a
        while W[j - 1] == W[a - 1]:
            j = j + 1

        b = j - 1
        for k in range(a, b + 1):
            T[k - 1] = (a + b) / 2

        i = b + 1

    for i in range(0, M):
        c = I[i]
        T_z[c] = T[i]

    return T_z


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

# taken from https://github.com/Thiagodcv/FastDeLong/blob/master/fast_delong.py


def fast_delong(X, Y):
    """
    Calculates and returns estimator of AUC ``theta_hat``, and
    variance-covariance matrix estimator ``S`` for ``theta_hat``.
    Parameters
    ----------
    X: (mxk) ndarray
        An array where X[i,:] = the probability estimates from all k models for AD-positive subject i,
        and X[:,j] = the probability estimates for all AD-positive subjects from model j
    Y: (nxk) ndarray
        An array where Y[i,:] = the probability estimates from all k models for control subject i,
        and Y[:,j] = the probability estmimates for all control subjects from model j
    Returns
    -------
    (kxk) ndarray
        The variance-covariance estimator ``S``
    (kx1) ndarray
        The AUC estimator ``theta_hat``
    """

    m = X.shape[0]
    n = Y.shape[0]
    k = X.shape[1]

    Z = np.concatenate((X, Y), axis=0)

    T_z = np.zeros((m + n, k))
    T_x = np.zeros((m, k))
    T_y = np.zeros((n, k))

    for r in range(0, k):
        T_z[:, r] = calculate_midranks(Z[:, r])
        T_x[:, r] = calculate_midranks(X[:, r])
        T_y[:, r] = calculate_midranks(Y[:, r])

    V_10 = (T_z[:m, :] - T_x) / n
    theta_hat = np.sum(T_z[:m, :], axis=0)
    theta_hat = theta_hat / (m * n) - (m + 1) / (2 * n)
    V_01 = 1 - (T_z[m:, :] - T_y) / m

    S_10 = np.zeros((k, k))
    S_01 = np.zeros((k, k))
    S = np.zeros((k, k))

    for r in range(0, k):
        for s in range(0, k):

            for i in range(0, m):
                S_10[r, s] = S_10[r, s] + (V_10[i, r] - theta_hat[r]) * (V_10[i, s] - theta_hat[s])

            for j in range(0, n):
                S_01[r, s] = S_01[r, s] + (V_01[j, r] - theta_hat[r]) * (V_01[j, s] - theta_hat[s])

            S[r, s] = S_10[r, s] / (m - 1) / m + S_01[r, s] / (n - 1) / n

    return S, theta_hat


def get_test_statistic(S, theta_hat):
    """
    Returns the test statistic of the Delong test. Note that the null hypothesis
    is set to L * theta = 0, where L = [1, -1]. This is equivalent to
    theta[0] - theta[1] = 0.
    Parameters
    ----------
    S: (kxk) ndarray
        The variance-covariance estimator ``S``
    theta_hat: (kx1) ndarray
        The AUC estimator ``theta_hat``
    Returns
    -------
    float
        The test statistic
    """

    L = np.array([1, -1])
    t = (L.T @ theta_hat) / (np.sqrt(L.T @ S @ L))  # This is standard normal
    return t


def delong_test(X, Y, test_type, alpha):
    """
    Performs the DeLong test on two different models.
    To specifiy the classification algorithm, the 2 files containing the 2 sets of classification results,
    and the path to the files, modify the global variables ``MODEL``,  ``FILENAMES``, and ``PATH`` respectively.
    Parameters
    ----------
    test_type: str
        Can be set to 'lower', 'upper', or 'two-tailed' for lower-tail, upper-tail, and two-tailed
        hypothesis testing respectively
    alpha: float
        The significance value, or the probability of rejecting the null hypothesis when it is true

    Returns
    -------
    dict
        Of the form {'T':t, 'p':p, 'lower_CI':lci, 'upper_CI':uci} where t is the test statistic, p is the p-value,
        lci is the lower confidence interval, and uci is the upper confidence interval
    """

    S, theta_hat = fast_delong(X, Y)
    t = get_test_statistic(S, theta_hat)

    # additional parameters for calculating confidence intervals.
    stdev = np.sqrt(S[0, 0] + S[1, 1] + 2 * S[0, 1])
    theta_dif = theta_hat[1] - theta_hat[0]

    if test_type == 'lower':
        p = norm.cdf(t)
        lci = theta_dif + norm.ppf(alpha) * stdev
        uci = None
    elif test_type == 'upper':
        p = 1 - norm.cdf(t)
        lci = None
        uci = theta_dif - norm.ppf(alpha) * stdev
    elif test_type == 'two-tailed':
        p = 2 * min(norm.cdf(t), 1 - norm.cdf(t))
        lci = theta_dif + norm.ppf(alpha / 2) * stdev
        uci = theta_dif - norm.ppf(alpha / 2) * stdev
    else:
        print('Wrong test_type!')

    return {'T': t, 'p': p, 'lower_CI': lci, 'Upper_CI': uci}

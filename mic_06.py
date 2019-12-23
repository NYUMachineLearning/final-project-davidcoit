import pyitlib as itl
from pyitlib import discrete_random_variable as drv
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import grangercausalitytests as granger
from matplotlib import pyplot
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from scipy.special import logsumexp as sp_logsumexp
import scipy.misc
import scipy.signal as sgn
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy import stats
from scipy import fftpack
from scipy import linalg
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import numba
import shutil
import os
import random
example = "06"


#from statsmodels.sandbox.distributions.mv_measures import mutual_info_kde


def mutualinfo_kde(y, x, normed=True):
    '''mutual information of two random variables estimated with kde
    '''
    nobs = len(x)
    if not len(y) == nobs:
        raise ValueError('both data arrays need to have the same size')
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    yx = np.vstack((y, x))
    kde_x = gaussian_kde(x)(x)
    kde_y = gaussian_kde(y)(y)
    kde_yx = gaussian_kde(yx)(yx)

    mi_obs = np.log(kde_yx) - np.log(kde_x) - np.log(kde_y)
    mi = mi_obs.sum() / nobs
    if normed:
        mi_normed = np.sqrt(1. - np.exp(-2 * mi))
        return mi_normed
    else:
        return mi

    # Define functions
# import transfer entropy function from R:

# calculate distance matrix from position matrix


def distanceMatrix(plist):
    n = np.shape(plist)[0]
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dmat[i, j] = np.linalg.norm(POSITIONS[i]-POSITIONS[j])
    return(dmat)

# calculate undirected adjacency matrix


def connectionMatrix(fmat, nlist):
    n = np.shape(fmat)[1]
    cmat = np.zeros((n, n))
    for connection in nlist:
        nfrom = connection[0] - 1
        nto = connection[1] - 1
        if connection[2] != 0:
            cmat[nfrom, nto] = 1
            cmat[nto, nfrom] = 1
    return(cmat)

# calculate directed adjacency matrix


def projectionMatrix(fmat, nlist):
    n = np.shape(fmat)[1]
    pmat = np.zeros((n, n))
    for connection in nlist:
        nfrom = connection[0] - 1
        nto = connection[1] - 1
        ctype = connection[2]
        pmat[nfrom, nto] = ctype
    return(pmat)


def spike_detect(flr, h=0.125, w=3):
    s = np.zeros(np.shape(flr))
    for j in range(np.shape(flr)[1]):
        t = sgn.find_peaks(flr[:, j],
                           h,
                           width=w)
        for i in t[0]:
            s[i, j] = 1
    return(s)


def lp_1(track):
    # attempt hp on 2d matrix
    try:
        new_track = np.zeros((np.shape(track)[0], np.shape(track)[1]))
        for i in range(np.shape(track)[0]):
            for j in range(np.shape(track)[1]):
                try:
                    new_track[i, j] = (track[i-1, j] + track[i, j] + track[i+1], j)
                except:
                    new_track[i, j] = track[i, j]

    # if dimensional error, hp 1 dimensional
    except:
        new_track = np.zeros(len(track))
        for i in range(len(track)):
            try:
                new_track[i] = (track[i-1] + track[i] + track[i+1])
            except:
                new_track[i] = track[i]
    return(new_track)


def lp_2(track):
    try:  # attempt hp on 2d matrix
        new_track = np.zeros((np.shape(track)[0], np.shape(track)[1]))
        for i in range(np.shape(track)[0]):
            for j in range(np.shape(track)[1]):
                try:
                    new_track[i, j] = (0.4 * track[i-3, j] + 0.6*track[i-2, j] +
                                       0.8*track[i-1, j] + track[i, j])
                except:
                    new_track[i, j] = track[i, j]
    except:  # if dimensional error, hp 1 dimensional
        new_track = np.zeros(len(track))
        for i in range(len(track)):
            try:
                new_track[i] = (0.4 * track[i-3] + 0.6*track[i-2] + 0.8*track[i-1] + track[i])
            except:
                new_track[i] = track[i]
    return(new_track)


def hp(track):
    try:  # attempt lp on 2d matrix
        new_track = np.zeros((np.shape(track)[0], np.shape(track)[1]))
        for i in range(np.shape(track)[0]):
            for j in range(np.shape(track)[1]):
                try:
                    new_track[i, j] = track[i, j] - track[i-1, j]
                except:
                    new_track[i, j] = track[i, j]
    except:
        new_track = np.zeros(len(track))
        for i in range(len(track)):
            try:
                new_track[i] = track[i] - track[i-1]
            except:
                new_track[i] = track[i]
    return(new_track)


def weight_filt(track):
    new_track = new_track = np.zeros((np.shape(track)[0], np.shape(track)[1]))
    for i in range(np.shape(track)[0]):
        for j in range(np.shape(track)[1]):
            new_track[i, j] = (track[i, j] + 1) ** (1 + (1 / np.sum(track[i, :])))
    return(new_track)


def thresh(track, th=0.05):
    try:  # attempt threshold on 2d matrix
        new_track = np.zeros((np.shape(track)[0], np.shape(track)[1]))
        for i in range(np.shape(track)[0]):
            for j in range(np.shape(track)[1]):
                if track[i, j] < th:
                    new_track[i, j] = 0
                else:
                    new_track[i, j] = track[i, j]
    except:  # threshold 1d track
        new_track = np.zeros(len(track))
        for i in range(len(track)):
            if track[i] < th:
                new_track[i] = 0
            else:
                new_track[i] = track[i]
    return(new_track)


def all_filter(track):
    track = weight_filt((thresh(lp(hp_2(hp_1(track))))))
    return(track)


def pc(mat):
    p = np.zeros((np.shape(mat)[1], np.shape(mat)[1]))
    prec = linalg.inv(np.cov(mat, rowvar=False))
    for i in range((np.shape(p)[0])):
        for j in range((np.shape(p)[1])):
            p[i, j] = -1 * prec[i, j] / (np.sqrt(prec[i, i]*prec[j, j]))
    return(p)


def mutual_info(matrix, window=50):
    n = np.shape(matrix)[1]
    MImat = np.zeros((n, n))
    periods = []
    for i in range(0, int(np.floor(np.shape(matrix)[0] / window)-1)):
        periods.append([i*window, (i+1)*window])

    for i in range(0, n):
        for j in range(0, n):
            MI = []
            for p in periods:
                #print("i={}, j={},p={}".format(i,j,p))
                if i != j:
                    info = mutual_info_classif(matrix[p[0]:p[1], i].reshape(-1, 1),
                                               matrix[p[0]:p[1], j],
                                               n_neighbors=5,
                                               discrete_features=True)*1000
                    MI.append(info)
                else:
                    MI.append(0)

                #print("{},{}: {}".format(i,j,MI))
            MImat[i, j] = (np.quantile(MI, .75))
    return(MImat)


def grangerMatrix(matrix, mem=5):
    # pass a spike train matrix
    n = np.shape(matrix)[1]
    gMat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                f = []
                c = matrix[:, (i, j)]
                g = granger(c, maxlag=mem, verbose=False)
                for k in g.keys():
                    f.append(g[k][0]['ssr_ftest'][0])
                gMat[i, j] = max(f)
    return(gMat)


def upper_thresh(matrix, th=15):
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if matrix[i, j] > th:
                matrix[i, j] = th
    return(matrix)


def mutual_information_conditional(matrix):
    n = np.shape(matrix)[1]
    MIC = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                print("{},{}".format(i, j))
                MIC[i, j] = drv.information_mutual_conditional(SPK_FLR[:, i],
                                                               SPK_FLR[:, j],
                                                               np.mean(np.delete(SPK_FLR[:, :],
                                                                                 [i, j],
                                                                                 axis=1),
                                                                       axis=1))
    return(MIC)


proj_dir = "/gpfs/scratch/dmc421/ML/connect/"
src_dir = "{}src/".format(proj_dir)
data_dir = "{}data/".format(proj_dir)
mat_dir = "{}mat/".format(proj_dir)

ffile = "flr_{}.txt".format(example)
nfile = "net_{}.txt".format(example)
pfile = "pos_{}.txt".format(example)

fpath = "{}{}".format(src_dir, ffile)
npath = "{}{}".format(src_dir, nfile)
ppath = "{}{}".format(src_dir, pfile)


# spike detection
SPK_FLR = np.genfromtxt("{}SPIKES_{}.csv".format(src_dir, example), delimiter=",")

MIC = mutual_information_conditional(SPK_FLR[0:1000, :])

np.savetxt("{}mutualconditionalinfo_{}.csv".format(mat_dir, example), MIC, delimiter=",")

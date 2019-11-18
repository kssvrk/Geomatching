# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:22:22 2019

@author: Rohit Gandikota (NR02440)
"""
import cv2
import numpy as np
import itertools
import sys

#%% Sample data
im = np.zeros((200,200))
im[10:60, 10:60] = 1

ref = np.zeros((500,500))
ref[50:110, 250:310] = 1

#%% SIFT and key point matching between image and reference
def findKeyPoints(im, ref, distance=200):
    detector = cv2.SIFT()
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    im_kp = detector.detect(im)
    im_kp, im_d = descriptor.compute(im, im_kp)

    ref_kp = detector.detect(ref)
    ref_kp, ref_d = descriptor.compute(ref, ref_kp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(im_d, flann_params)
    idx, dist = flann.knnSearch(ref_d, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    im_kp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            im_kp_final.append(im_kp[i])

    flann = cv2.flann_Index(ref_d, flann_params)
    idx, dist = flann.knnSearch(im_d, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    ref_kp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            ref_kp_final.append(ref_kp[i])

    return im_kp_final, ref_kp_final
#%% test
ikp, rkp = findKeyPoints(im, ref)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:09:16 2018

@author: coelhorp
"""

from pyriemann.utils.base import powm, sqrtm, invsqrtm
from pyriemann.utils.distance import distance_riemann

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax

import numpy as np

def power_means(C, p):
    phi = 0.375/np.abs(p)    
    K = len(C)
    n = C[0].shape[0]
    w = np.ones(K)
    w = w/(1.0*len(w))
    G = np.sum([wk*powm(Ck, p) for (wk,Ck) in zip(w,C)], axis=0)
    if p > 0:
        X = invsqrtm(G)    
    else:
        X = sqrtm(G)    
    zeta = 10e-10
    test = 10*zeta
    while test > zeta:
        H = np.sum([wk*powm(np.dot(X, np.dot(powm(Ck, np.sign(p)), X.T)), np.abs(p)) for (wk,Ck) in zip(w,C)], axis=0)   
        X = np.dot(powm(H, -phi), X)
        test = 1.0/np.sqrt(n) * np.linalg.norm(H - np.eye(n))
    if p > 0:
        P = np.dot(np.linalg.inv(X), np.linalg.inv(X.T))    
    else:
        P = np.dot(X.T, X)        
    return P
   
class MeanFieldClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Mean fields classifier
    """        
    def __init__(self, plist=[-1,+1]):
        """Init."""
        self.plist = plist

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.
        """
        self.classes_ = np.unique(y)      
        
        def means_field_train(covs, labs, plist):
            means = {}
            for p in plist:
                means_p = {}
                for label in np.unique(labs):
                    means_p[label] = power_means(covs[labs == label], p)
                means[p] = means_p    
            return means        
        
        self.covmeans_ = means_field_train(X, y, self.plist)

        return self 

    def predict(self, covtest):
        """get the predictions.
        """        

        def means_field_test(covs, plist, means_train):
            labs = sorted(means_train[plist[0]].keys())
            labs_pred = []
            for covi in covs:
                m = {}
                for p in plist:       
                    m[p] = []
                    for label in np.unique(labs):
                        m[p].append(distance_riemann(covi, means_train[p][label])**2)
                pmin = min(m.items(), key=lambda x: np.sum(x[1]))[0]
                yi = np.unique(labs)[np.argmin(m[pmin])]
                labs_pred.append(yi)
            labs_pred = np.array(labs_pred)    
            return labs_pred           
        
        return means_field_test(covtest, self.plist, self.covmeans_)
    
    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform.""" 

        dist = []
        for covi in covtest:         
            m = {}
            for p in self.plist:       
                m[p] = []
                for label in self.classes_:
                    m[p].append(distance_riemann(covi, self.covmeans_[p][label])**2)
            pmin = min(m.items(), key=lambda x: np.sum(x[1]))[0]                
            dist.append(np.array(m[pmin]))
                        
        return np.stack(dist)    

    def transform(self, X):
        """get the distance to each centroid.
        """
        return self._predict_distances(X)    

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)
    
    def predict_proba(self, X):
        """Predict proba using softmax.
        """
        return softmax(-self._predict_distances(X))     
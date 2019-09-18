#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:32:42 2018

@author: coelhorp
"""

import numpy as np
from tqdm import tqdm

from pyriemann.classification import MDM

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from pyriemann.estimation import Covariances

from riemann_lab import power_means
import moabb
moabb.set_log_level('info')
from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

# define the dataset, the paradigm, and the evaluation procedure
datasets = [BNCI2014001()]
paradigm = LeftRightImagery()
evaluation = WithinSessionEvaluation(paradigm, datasets)

# define the pipelines for classification -- MDM and MeansField classifier
pipelines = {}
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
plist = [1.00, 0.75, 0.50, 0.25, 0.10, 0.01, -0.01, -0.10, -0.25, -0.50, -0.75, -1.00]
pipelines['MeansField'] = make_pipeline(Covariances('oas'), power_means.MeanFieldClassifier(plist=plist))

# loop through the subjects and datasets
results = evaluation.process(pipelines)



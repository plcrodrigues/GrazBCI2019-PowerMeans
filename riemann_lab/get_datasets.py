#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:33:44 2018

@author: coelhorp
"""

from sklearn.externals import joblib
from scipy.io import loadmat
import numpy as np

def get_dataset(dataset, subject, session, storage='GIPSA'):
    '''
    Generic function that gets the dataset from a folder
    The options of datasets are (call them via the keyword on the list below)
        MotorImagery : AlexMI
                       BNCI2014001
                       BNCI2014002
                       BNCI2014004
                       BNCI2015001
                       Cho2017
                       MunichMI
                       Ofner2017
                       PhysionetMI
                P300 : P300-BrainInvaders
                       P300-Erwan
               SSVEP : SSVEP
           Simulated : Simulated
    Note that the data stored in MotorImagery and P300 are in the form of 'signals' and 'labels'
    Whereas for SSVEP and Simulated we have only 'covs' and 'labels'
    '''

    # if working at GIPSA-lab, the datasets are stored at the /research/vibs server
    if storage == 'GIPSA':
        path = '/research/vibs/Pedro/datasets/moabb/'
    # otherwise, pass the path to the folder containing all datasets
    else:
        path = storage

    path = path + dataset + '/'
    filename = 'subject_' + str(subject) + '_' + session + '.pkl'
    data = joblib.load(path + filename)

    return data

def get_dataset_physionet_covs(subject, full=False, storage='GIPSA'):
    '''Get the .pkl containing all Covariance matrices estimated from the signals of the PhysionetMI dataset
       We have the option of taking the Covariance matrices from all electrodes or from a few selected ones
       idx = [31,33,35,2,4,8,10,12,16,18,48,52] (starts counting at zero)
       idx = [F3,Fz,F4,FC1,FC2,C3,Cz,C4,CP1,CP2,P3,P4]
    '''

    # if working at GIPSA-lab, the datasets are stored at the /research/vibs server
    if storage == 'GIPSA':
        path = '/research/vibs/Pedro/datasets/moabb/PhysionetMI'
    # otherwise, pass the path to the dataset folder
    else:
        path = storage

    if full:
        filename = path + '/covsfull/subject_' + str(subject) + '_full.pkl'
    else:
        filename = path + '/covsredu/subject_' + str(subject) + '.pkl'
    data = joblib.load(filename)

    return data

def get_dataset_gigadb_covs(subject, full=False, storage='GIPSA'):
    '''Get the .pkl containing all Covariance matrices estimated from the signals of the GigaDB dataset
       We have the option of taking the Covariance matrices from all electrodes or from a few selected ones
       idx = [4,37,39,8,10,45,43,13,12,47,49,50,16,17,18,55,54,53,20,30,57,25,62] (starts counting at zero)
       idx = [F3,Fz,F4,FC5,FC1,FC2,FC6,C5,C3,Cz,C4,C6,CP5,CP3,CP1,CP2,CP4,CP6,P3,Pz,P4,PO3,PO4]
    '''

    # if working at GIPSA-lab, the datasets are stored at the /research/vibs server
    if storage == 'GIPSA':
        path = '/research/vibs/Pedro/datasets/moabb/Cho2017'
    # otherwise, pass the path to the dataset folder
    else:
        path = storage

    if full:
        filename = path + '/covsfull/subject_' + str(subject) + '_full.pkl'
    else:
        filename = path + '/covsredu/subject_' + str(subject) + '.pkl'
    data = joblib.load(filename)

    covs = data['covs']
    labs = data['labels']

    # reorganize the trials in a sequence manner according to the experimental runs
    filename = path + '/trial_sequence/s' + str(subject) + '_trial_sequence_v1.mat'
    sequence = loadmat(filename)
    L_hand_sequence = sequence['trial_sequence'][0][0][0].squeeze()
    R_hand_sequence = sequence['trial_sequence'][0][0][1].squeeze()

    L_hand_sequence = np.array([(ind, 0, i) for i, ind in enumerate(L_hand_sequence)])
    R_hand_sequence = np.array([(ind, 1, i+100) for i, ind in enumerate(R_hand_sequence)])
    hand_sequence = np.concatenate((L_hand_sequence, R_hand_sequence))
    idx = hand_sequence[:,0].argsort()
    hand_sequence = hand_sequence[idx]

    data = {}
    data['covs'] = covs[idx]
    data['labels'] = labs[idx]

    return data

def get_settings(dataset, storage):

    settings = {}
    settings['storage'] = storage

    if dataset == 'AlexMI':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8]
        settings['ncovs_list'] = [1, 5, 10, 15]

    elif dataset == 'BNCI2014001':
        settings['dataset'] = dataset
        settings['session'] = 'session_E'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        settings['ncovs_list'] = [1, 6, 18, 36]

    elif dataset == 'BNCI2014002':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14] # numerical problems on subject 8
        settings['ncovs_list'] = [1, 5, 10, 20]

    elif dataset == 'BNCI2015001':
        settings['dataset'] = dataset
        settings['session'] = 'session_A'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        settings['ncovs_list'] = [1, 5, 10, 25]

    elif dataset == 'BNCI2015004':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        settings['ncovs_list'] = [1, 5, 10, 20]

    elif dataset == 'PhysionetMI':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = range(1, 109+1)
        settings['ncovs_list'] = [1, 5, 10, 15]

    elif dataset == 'Cho2017':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = list(range(1, 52+1))
        for ii in [32, 46, 49]:
            settings['subject_list'].remove(ii)
        settings['ncovs_list'] = [1, 5, 10, 25]

    elif dataset == 'MunichMI':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        settings['ncovs_list'] = [1, 25, 50, 75]

    elif dataset == 'Ofner2017':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'MI'
        settings['subject_list'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        settings['ncovs_list'] = [1, 5, 20, 40]

    elif dataset == 'SSVEP':
        settings['dataset']  = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'SSVEP'
        settings['subject_list'] = range(1, 12+1)
        settings['ncovs_list'] = [1, 2, 4, 6]

    elif dataset == 'P300-Erwan':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'P300'
        settings['subject_list'] = range(1, 24+1)
        settings['ncovs_list'] = [6, 12, 32, 48]

    elif dataset == 'P300-BrainInvaders':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'P300'
        settings['subject_list'] = list(range(1, 48+1))
        for ii in [2, 7, 8, 16, 17, 20, 27, 29, 32, 34, 38, 40, 45, 47]:
            settings['subject_list'].remove(ii)
        settings['ncovs_list'] = [5, 10, 20, 30]

    elif dataset == 'Simulated':
        settings['dataset'] = dataset
        settings['session'] = 'session_0'
        settings['paradigm'] = 'Simulated'
        settings['subject_list'] = [1, 2]
        settings['ncovs_list'] = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

    return settings


from sklearn.model_selection import StratifiedKFold
import numpy as np

def score_cross_validation(clf, data, n_splits=10):

    covs = data['covs']
    labs = data['labels']

    scores = []

    kf = StratifiedKFold(n_splits=n_splits)

    for train_idx, test_idx in kf.split(covs, labs):
        covs_train, labs_train = covs[train_idx], labs[train_idx]
        covs_test, labs_test = covs[test_idx], labs[test_idx]
        clf.fit(covs_train, labs_train)
        scores.append(clf.score(covs_test, labs_test))

    return np.mean(scores), np.std(scores)

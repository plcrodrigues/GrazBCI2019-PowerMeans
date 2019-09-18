
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def permutations_paired_t_test(df, nrzt, multiple_correction=True):

    '''
    Perform a permutations based t-test on the input data
    We do all pairwise comparisons and may adjust the pvalues via Holm's stepdown procedure
    The input dataframe has the accuracies of each method on each column
    The output dataframe contains the pvalues for each comparisons and also the signs of the t-test
    '''

    # how many paired measurements we have
    ninstances = len(df)

    # get all the pairwise comparisons to do
    pairs = list(combinations(df.columns, 2))
    pairs_names = [pair[1] + '-' + pair[0] for pair in pairs]

    # getting the observed values for the statistics and the differences between accuracies on pairs
    df_pairs = {}
    df_pairs_statistic_observed = {}
    df_pairs = pd.DataFrame()
    for pair_name, pair in zip(pairs_names, pairs):
        df_pairs[pair_name] = df[pair[1]] - df[pair[0]]
    df_pairs_statistic_observed = np.sum(df_pairs, axis=0)

    # initialize the counter variables
    counters = pd.DataFrame(np.zeros((1, len(pairs))), columns=pairs_names)

    # perform the permutations by multiplying the differences by -1 or +1 randomly
    for _ in range(nrzt):
        perms = pd.DataFrame(-1+2*np.random.randint(0,2,size=(ninstances, len(pairs))), columns=pairs_names)
        df_pairs_statistic_permutation = np.sum(df_pairs.multiply(perms), axis=0)
        counters = counters + (np.abs(df_pairs_statistic_observed) < np.abs(df_pairs_statistic_permutation))
    counters = (counters.T + 1) / (nrzt + 1)

    # get the statistics
    stats = pd.DataFrame()
    stats['sign'] = np.sign(df_pairs_statistic_observed)
    stats['pvalue'] = counters

    # correct the pvals via Holm's method with Sidak's corrections
    _,pvals_corrected,_,_ = multipletests(stats['pvalue'], alpha=0.05, method='holm-sidak')
    stats['pvalue'] = pvals_corrected

    return stats

from functools import reduce
import pandas as pd
import numpy as np

def casIn(cascade, time_decay = -0.000068, alpha = None, beta = 1.0):
    """Computes influence in one cascade
    
    Parameters
    ----------
    cascade : str or DataFrame
        Path to one cascade in a file
    time_decay : float
        The r parameter described in the paper
    alpha : float, optional
        A float between 0 and 1, as described in the paper. If None DebateNight method is used, else spatial-decay method, by default None
    
    Returns
    -------
    DataFrame
        A dataframe describing the influence of each user in a single cascade.
    """
    if isinstance(cascade, str):
        cascade = pd.read_csv(cascade_path) # Read one cascade from local file
    p_ij = P(cascade, alpha = alpha, r=time_decay, beta = beta) # compute p_ij in given cascade
    inf, m_ij = influence(p_ij, alpha) # compute user influence
    cascade["influence"] = pd.Series(inf)
    return cascade


def P(cascade, r = -0.000068, beta = 1.0):
    """Computes the P matrix of a cascade 
    
    The P matrix describes the stochastic retweet graph.
    
    Parameters
    ----------
    cascade : DataFrame
        A dataframe describing a single cascade, with a time column ascending from 0, a magnitude column and index of user ids
    r : float, optional
        The time-decay r parameter described in the paper, by default -0.000068
    beta : float, optional
        A social strength hyper-parameter, by default 1.0
    
    Returns
    -------
    array-like
        A matrix of size (n,n), where n is the number of tweets in the cascade, where P[i][j] is the probability that j is a retweet of tweet i.
    """    
    n = len(cascade)
    t = np.zeros(n,dtype = np.float64)
    f = np.zeros(n,dtype = np.float64)
    p = np.zeros((n,n),dtype = np.float64)
    norm = np.zeros(n,dtype = np.float64)
    for k, row in cascade.iterrows():
        if k == 0:
            p[0][0] = 1
            t[0] = row['time']
            if np.isnan(row['magnitude']):
                print(row)
            f[0] = 1 if row['magnitude'] == 0 else row['magnitude']
            continue
        
        t[k] = row['time']
        f[k] = (1 if row['magnitude'] == 0 else row['magnitude'])**beta
        p[:k, k] = ((r * (t[k] - t[0:k])) + np.log(f[0:k])) # store the P_ji in log space
        norm[k] = reduce(np.logaddexp, p[:k, k])
        p[:k, k] = np.exp(p[:k, k] - norm[k])# recover the P_ji from log space
    return p


def influence(p, alpha = None):
    """Estimates user influence

    This function compute the user influence and store it in matirx m_ij
    
    Parameters
    ----------
    p : array-like
        The P matrix describing the stochastic retweet graph
    alpha : float, optional
        A float between 0 and 1, as described in the paper. If None DebateNight method is used, else spatial-decay method, by default None
    
    Returns
    -------
    array-like, array-like
        A n-array describing the influence of n users/tweets and the (n,n)-array describing the intermediary contribution of influence between tweets
    """    
    p *= (alpha if alpha else 1)
    n = len(p)
    m = np.zeros((n, n))
    m[0, 0] = 1
    for i in range(0, n-1):
        vec = p[:i+1, i+1]
        m[:i+1, i+1] = m[:i+1, :i+1]@vec
        m[i+1, i+1] = (1-alpha if alpha else 1)
    influence = np.sum(m, axis = 1)

    return influence, m


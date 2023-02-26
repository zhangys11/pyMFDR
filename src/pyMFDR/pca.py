import numpy as np


def pca_raw(X, k=2, verbose=True):
    '''
    This is a raw implemention of PCA according to its math defintion. 
    Usually, we can just use sklearn.decomposition.PCA instead of this one.
    '''

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    # /sigma  # sigma contains 0. cause error : LinAlgError: SVD did not converge
    X_norm = (X - mu)

    if verbose:
        print("\n均值 mu = ")
        print(mu)

        print("\n标准差 sigma = ")
        print(sigma)

    # If rowvar is True (default), then each row represents a variable, with observations in the columns.
    # Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
    SIGMA = np.cov(X_norm, rowvar=False)

    U, s, V = np.linalg.svd(SIGMA)

    s = np.sort(s)[::-1]

    if verbose:

        print("\n特征值 s = ")
        print(s)

        print("\n特征向量 U = ")
        print(U)

        print("\n相关系数矩阵（corrcoef） = ")
        print(np.corrcoef(X_norm.T))

    i = 1
    accumulated = 0
    for c in s[0:5]:
        accumulated = accumulated + c

        if verbose:
            print('第{}成分的方差百分比 = {}, 累积方差百分比 = {}'.format(
                i, c/np.sum(s), accumulated/np.sum(s)))

        i = i+1

    X_compressed = np.dot(X, U[:, :k])

    return X_compressed, mu, sigma, U

from pyDRMetrics.pyDRMetrics import *
from pyDRMetrics.calculate_recon_error import calculate_recon_error
from sklearn.decomposition import PCA, NMF, FastICA
from matplotlib.pyplot import MultipleLocator
import time
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from tqdm import tqdm
from sklearn import random_projection
import scipy.stats as stats
from sklearn.cluster import KMeans
from scipy.sparse.csr import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from .archetypes import ArchetypalAnalysis
from .lae import *

#import imp
#imp.reload(lae) # only use in debug / dev mode

# sum_bin and tri_bin are also MF DR algs, where H is 
# [[111100000000],[000011110000],[000000001111]] or 
# [[123210000000000],[000001232100000],[000000000012321]].
# max_bin is adaptive, depending on the index of max element.
ALGS = ['PCA','NMF','LAE','RP','SRP','VQ','AA','ICA'] 


def get_algorithms():
    return ALGS

def mf(X, k, alg = 'PCA', display = True, verbose = 0):
    '''
    A static method that performs specified MF algorithm.
    
    Parameters
    ----------
    X : the original data. m-by-n matrix.
    k : target dimensionality
    alg : 'PCA', 'NMF', 'VQ', etc.
    display : whether output algorithm-specific chart/diagram
    verbose: set verbose level for deep learning-based algorithms, e.g., autoencoder.

    Returns
    -------
    W : data after DR. m-by-k matrix. Typically, k << n
    H : dictionary. k-by-n matrix.
    Xr : reconstructed data. m-by-n matrix.
    o : the inner algorithm instance
    '''
    if (alg == ALGS[0]): # in statistics, PCA is a factor extraction method for FA. 
        pca = PCA(n_components = k, svd_solver='arpack') # ARPACK is a collection of Fortran77 subroutines designed to solve large scale eigenvalue problems.
        W = pca.fit_transform(X)
        Xr = pca.inverse_transform(W)
        H = pca.components_
        o = pca
        
        if display:
            plt.figure(figsize=(10,3))
            plt.scatter(list(range(1,k+1)), pca.explained_variance_ratio_, alpha=0.7, label='variance percentage')
            plt.scatter(list(range(1,k+1)), pca.explained_variance_ratio_.cumsum(), alpha=0.5, label='cumulated variance percentage')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
            
    elif (alg == ALGS[1]):

        # solver: default solve is cd (coordinate descent), the other is mu (Multiplicative Update solver). sd is faster than mu.
        # init: NNDSVDa（全部零值替换为所有元素的平均值）和 NNDSVDar（零值替换为比数据平均值除以100 小的随机扰动）
        nmf = NMF(n_components=k, init='nndsvdar', shuffle = True)
        W = nmf.fit_transform(X)
        H = nmf.components_ 
        Xr = nmf.inverse_transform(W)
        o = nmf
        
        if display:
            
            errors = []
            r = range(1, min(15, X.shape[1]) ,1)
            for k in r:
                nmf = NMF(n_components=k, init='nndsvdar', shuffle = True)
                nmf.fit(X)
                errors.append(nmf.reconstruction_err_)    
                
            plt.figure(figsize=(10,3))
            plt.scatter(r, errors, alpha = .7, label = 'error')
            plt.scatter(r, np.gradient(errors), alpha = .5, label = 'gradient of error')
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
            
    elif (alg == ALGS[2]):
        
        '''
        An autoencoder, is an artificial neural network used for learning efficient codings.
        The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.
        An autoencoder is unsupervised since it's not using labeled data. The goal is to minimize reconstruction error based on a loss function, such as the mean squared error:

        $\mathcal{L}(\mathbf{x},\mathbf{x'})=\|\mathbf{x}-\mathbf{x'}\|^2=\|\mathbf{x}-f(\mathbf{W'}(f(\mathbf{Wx}+\mathbf{b}))+\mathbf{b'})\|^2$

        AutoEncoder (as well as other NN models, such as MLP) is sensitive to feature scaling, so it is highly recommended to scale your data.
        '''

        ae = LAE(n_components=k)
        W = ae.fit_transform(X, verbose = verbose)
        H = ae.components_ 
        Xr = ae.inverse_transform(W)
        o = ae
        
        if display:
            
            LOSS1 = []
            LOSS2 = []
            r = list(range(1, 11))
            for D in r:

                l1 = 0 # 0.001
                l2 = 0.01
                lae = LAE(D)
                lae.fit(X, epochs = 1000, batch_size = 4, l1_reg = l1, l2_reg = l2)
                # encoder, decoder, ae, hist = build_1_linear_dense_layer_auto_encoder(X_scaled, encoding_dim = D, epochs = 1000, batch_size = 4, l1_reg = l1, l2_reg = l2)
                LOSS1.append(lae.hist.history['loss'][-1])
                LOSS2.append(lae.hist.history['val_loss'][-1])

            plt.figure(figsize=(10,3))
            plt.scatter(r, LOSS2, alpha = .7, label = 'val loss')
            plt.scatter(r, np.gradient(LOSS2), alpha = .5, label = 'gradient of val loss')
            #plt.plot(r, LOSS1, alpha = .7)
            #plt.plot(r, np.gradient(LOSS2), alpha = .5)
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
            
    elif (alg == ALGS[3] or alg == 'GRP' or alg == ALGS[4]):
        
        rp = random_projection.GaussianRandomProjection(n_components = k) # if unspecified, k will be auto generated by Johnson-Lindenstrauss lemma, at eps = 0.1.
        if (alg == 'SRP'):
            rp = random_projection.SparseRandomProjection(n_components = k)
        
        W = rp.fit_transform(X)
        H = rp.components_
        Xr = W @ rp.components_ # use W @ H to approximate X
        o = rp           
        
        if display:
            
            AUCs = []
            Qlocals = []
            r= list(range(1,20))
            
            for k in r:
                rp = random_projection.GaussianRandomProjection(n_components = k)
                if (alg == 'SRP'):
                    rp = random_projection.SparseRandomProjection(n_components = k)
                X_rp = rp.fit_transform(X)
                Xr_rp = X_rp @ rp.components_
                drm = DRMetrics(X, X_rp, Xr_rp)
                AUCs.append(drm.AUC)
                Qlocals.append(drm.Qlocal)
            
            plt.figure(figsize=(10,3))
            plt.scatter(r, AUCs, alpha = .7, label = 'AUC')
            plt.scatter(r, Qlocals, alpha = .5, label = 'Qlocal')
            plt.plot(r, AUCs, alpha = .7)
            plt.plot(r, Qlocals, alpha = .5)
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
    
    elif (alg == ALGS[5]):

        kmeans = KMeans(n_clusters=k)

        # W = kmeans.fit_transform(X) # NOTE: transform() returns distance matrix from centroids.       
        W = kmeans.fit_predict(X) # returns class id array
        W = np.eye(k)[W] # convert W to one-hot encoding matrix

        H = kmeans.cluster_centers_
        #ohe = OneHotEncoder()
        #W = ohe.fit_transform(W.reshape(-1, 1)).A
        
        o = kmeans
        Xr = W @ H
        
        if display:
            
            Scores = []
            r = list(range(1, 20))

            for k in r:
                kmeans = KMeans(n_clusters=k).fit(X)
                Xvq = kmeans.predict(X)
                Scores.append(kmeans.score(X, Xvq))

            plt.figure(figsize=(10,3))
            plt.scatter(r, Scores, alpha = .7, label = 'score')
            plt.scatter(r, np.gradient(Scores), alpha = .5, label = 'score gradient')
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
        
    elif (alg == ALGS[6]):
        aa = ArchetypalAnalysis(n_archetypes = k) # tolerance = 0.001, max_iter = 200, C = 0.0001, initialize = 'random', redundancy_try = 30
        aa.fit(X)
        H = aa.archetypes.T
        W = aa.alfa.T
        Xr = W @ H #(aa.alfa.T) @ (aa.archetypes.T)
        o = aa
        
        if display:
            
            lst_exp_var = []
            r = range(1, min(11, X.shape[1]) ,1)

            for k in r:
                aa = ArchetypalAnalysis(n_archetypes = k)
                aa.fit(X)
                lst_exp_var.append(aa.explained_variance_)

            plt.figure(figsize=(10,3))
            plt.scatter(r, lst_exp_var, alpha=0.5, label='cumulated variance')
            plt.scatter(r, np.gradient(lst_exp_var), alpha = .5, label = 'gradient of cumulated variance')
            plt.legend(loc="upper right")
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()
    
    elif (alg == ALGS[7]):
        ica = FastICA(n_components=k, max_iter=1000, tol=0.1)
        W = ica.fit_transform(X)
        Xr = ica.inverse_transform(W)
        H = ica.components_ # H kxn The linear operator to apply to the data to get the independent sources. This is equal to the unmixing matrix 
        # Hinv = ica.mixing_.shape # nxk The pseudo-inverse of components_. It is the linear operator that maps independent sources to the data. 
        o = ica
        
        if display:
            n_cs = list(range(1, 10))

            MSEs = []
            for n_c in n_cs:
                transformer = FastICA(n_components=n_c)
                X_transformed = transformer.fit_transform(X)
                Xr = transformer.inverse_transform(X_transformed)
                mse, ms, rmse = calculate_recon_error(X, Xr)
                MSEs.append(mse)

            plt.figure(figsize=(10,3))
            plt.scatter(n_cs, MSEs, alpha = .7, label = 'error')
            plt.scatter(n_cs, np.gradient(MSEs), alpha = .5, label = 'gradient of error')
            #plt.plot(r, LOSS1, alpha = .7)
            #plt.plot(r, np.gradient(LOSS2), alpha = .5)
            plt.xlabel('k')
            plt.legend()
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.show()            
    
    
    assert X.shape == Xr.shape
    assert X.shape[0] == W.shape[0]
    assert X.shape[1] == H.shape[1]
    assert W.shape[1] == H.shape[0]
    
    return W,H,Xr,o

def evaluate_dr(X,W,Xr):
    '''
    Evaluate the DR quality by multiple metrics. Provided by the pyDRMetric package.
    
    Parameters
    -------
    X : original data. m-by-n matrix.
    W : data after DR. m-by-k matrix. Typically, k << n
    Xr : reconstructed data. m-by-n matrix.
    
    Reference
    ---------
    pyDRMetrics - A Python toolkit for dimensionality reduction quality assessment, Heliyon, Volume 7, Issue 2, 2021, e06199, ISSN 2405-8440, https://doi.org/10.1016/j.heliyon.2021.e06199.
    '''
    drm = DRMetrics(X, W, Xr)
    drm.report()

def visualize_dictionary_distribution(H, dist = 'gassian'):
    '''
    Draw histogram of dictionary/components/loadings, along with theoreticl distribution PDF.
    
    Parameters
    -------
    H : dictionary/components/loadings. k-by-n matrix.
    dist : target distribution to fit/approach. default Gaussian.
    '''
    
    if isinstance(H, csr_matrix):
        H = H.A

    for i in range(len(H)):
        
        _, x, _ = plt.hist(H[i], density=True, histtype=u'step', bins=np.linspace(-3, 3, 80) )
        
        if dist == 'gaussian':
            density = stats.gaussian_kde(H[i])
            plt.plot(x, density(x))    
        
        plt.title('Component Loading ' + str(i+1))
        plt.show()

        print('mu = ' , round(np.mean(H[i]),3), 
                'std = ', round(np.std(H[i]),3))

def visualize_dictionary(H):
    '''
    Visualize the dictionary matrix.
    
    Parameters
    -------
    H : dictionary. k-by-n matrix.
    '''
    
    if isinstance(H, csr_matrix):
        H = H.A
    
    plt.figure(figsize = (100,25))

    for row in range(H.shape[0]):    
        h = H[row,:]   
        plt.plot(list(range(len(h))), h.tolist(), label= 'Component ' + str(row+1)) # scatter sometimes cannot correctly auto adjust y range

    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False) #.set_ticklabels([])
    cur_axes.axes.get_yaxis().set_visible(False) #.set_ticklabels([])       

    plt.legend(fontsize=50)
    plt.title(u'Dictionary Visualization\n', fontsize=50)
    plt.show()
    
def measure_time_all(X, repeat = 5):

    if X.min() < 0:
        print('Because some algorithms (e.g., NMF) require non-negative input, we shift up X by | X.min() |')
        X = X - X.min()

    TSS = {}
    for alg in ALGS:
        TSS[alg] = measure_time(X, alg, repeat = repeat)
    return TSS

def measure_time(X, alg = 'PCA', display = True, r = list(range(1, 21)), repeat = 5):
    '''
    Measure the time taken for specific MF algorithm. The time is averaged for multiple runs.
    
    Parameters
    ----------
    X
    alg : 'PCA', 'NMF', 'VQ', etc.
    display : whether output the time~k chart
    r : k range to be plotted
    repeat: how many runs to be averaged

    Returns
    -------
    TS : an array of consumed times for different k values.
    '''

    TS = []

    for k in r:
        T = 0
        ts = []
        for i in range(repeat):
            time1 = time.time_ns()    
            _ = mf(X, k, alg, False)
            time2 = time.time_ns()
            ts.append((time2-time1)/ (10 ** 6))

        if (repeat >= 5):
            ts.remove(max(ts))
            ts.remove(min(ts))
        TS.append( np.array(ts, np.float).mean() )

    if display:
        plt.figure(figsize=(10,3))
        plt.scatter(r, TS, alpha=0.7, label='used time (ms)')
        plt.plot(r, TS, alpha=0.7)
        plt.title('time consumption (' + alg + ') ~ k')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.show()

    return TS


def visualize_reconstruction(X, Xr, N):
    '''
    Visualize top-N samples before and after DR side by side.
    '''
    
    for idx in range(N):
        visualize_one_sample_reconstruction(X, Xr, idx)
    
def visualize_one_sample_reconstruction(X, Xr, idx = 0, figsize = (50,10)):
    '''
    Visualize the n-th(default 1st) sample before and after DR side by side.
    '''
    
    assert(X.shape == Xr.shape)
    assert(idx < X.shape[0])

    # original
    plt.figure(figsize=figsize)
    
    ax = plt.subplot(1, 1, 1)
    ax.scatter(list(range(X.shape[1])), list(X[idx]))        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    print("X[" + str(idx) + "]")
    plt.show()


    plt.figure(figsize=figsize)

    # reconstruction
    ax = plt.subplot(1, 1, 1)
    ax.scatter(list(range(Xr.shape[1])), list(Xr[idx]))        
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    print("Xr[" + str(idx) + "]")
    plt.show()

def compare_all(X, k = 3):
    '''
    Compare all MFDR algorithms
    '''

    if X.min() < 0:
        print('Because some algorithms (e.g., NMF) require non-negative input, we shift up X by | X.min() |')
        X = X - X.min()

    for alg in ALGS:
        
        print()
        print()
        print('======= ', alg , ' =======')
        print()
        
        W,H,Xr,o = mf(X, k, alg = alg, display = False)
        evaluate_dr(X,W,Xr)    
        visualize_dictionary(H)    
        visualize_one_sample_reconstruction(X, Xr) 
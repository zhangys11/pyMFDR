# pyMFDR

<b>publication:</b> <i>Matrix Factorization Based Dimensionality Reduction Algorithms - A Comparative Study on Spectroscopic Profiling Data. Analytical Chemistry, 2022, DOI: [10.1021/acs.analchem.2c01922](https://doi.org/10.1021/acs.analchem.2c01922)</i>

<hr/>
This is a python package for MF (matrix factorization) based DR (dimensionality reduction) algorithms.  

<pre>
  Content of repo
  ├── src : source code
  ├── data : contains the dataset (.csv) used for the study
  └── notebooks : contains the jupyter notebook for the study
</pre>

# Installation 

pip install pyMFDR

# How to use 

Download the sample dataset from the /data folder
Use the following sample code to use the package:

<pre>
  # import the library
  from pyMFDR import mfdr

  # load the dataset or generate a toy dataset by X,y = mvg(md = 2)
  df = pd.read_csv('7047_C02.csv')
  X = df.iloc[:,2:cols-1].values # -1 for removing the last column that contains NAN
  y = df.iloc[:,1].values.ravel() # first col is index and not used in this study

  # get a list of available MFDR algorithms
  mfdr.get_algorithms() # it will ouptut ['PCA', 'NMF', 'LAE', 'RP', 'SRP', 'VQ', 'AA', 'ICA']

  # Run PCA on X. It will return W, H, Xr and the inner algorithm object.
  W,H,Xr,o = mfdr.mf(X, 3, alg = 'PCA', display = False) 

  # evaluate the dimensionality reduction quality by various metrics
  mfdr.evaluate_dr(X,W,Xr)

  # visualize H
  mfdr.visualize_dictionary(H)

</pre>

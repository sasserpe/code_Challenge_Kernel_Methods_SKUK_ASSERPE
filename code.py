# -*- coding: utf-8 -*-
"""
# Kernel Methods Data Challenge

Authors:

Breno BALDAS SKUK & Samuel ASSERPE
"""

"""
# 1. Downloading Data
"""

import pandas as pd
import numpy as np
from scipy import linalg


# 1st dataset
df_train_labels_0 = pd.read_csv("data/Ytr0.csv")
train_labels_0 = np.ravel(df_train_labels_0['Bound'].to_numpy())

# 2nd dataset 
df_train_labels_1 = pd.read_csv("data/Ytr1.csv")
train_labels_1 = np.ravel(df_train_labels_1['Bound'].to_numpy())

# 3rd dataset
df_train_labels_2 = pd.read_csv("data/Ytr2.csv")
train_labels_2 = np.ravel(df_train_labels_2['Bound'].to_numpy())

print(f'Datasets labels imported')

# Load precomputed matrices

cols = [str(i) for i in range(100)] # create some col names

# 1st dataset
Xtr0_mat = pd.read_csv("data/Xtr0_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()
Xte0_mat = pd.read_csv("data/Xte0_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()

# 2nd dataset
Xtr1_mat = pd.read_csv("data/Xtr1_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()
Xte1_mat = pd.read_csv("data/Xte1_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()

# 3rd dataset
Xtr2_mat = pd.read_csv("data/Xtr2_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()
Xte2_mat = pd.read_csv("data/Xte2_mat100.csv", sep="\s+|;|:", names=cols, header=None, engine="python").to_numpy()

print(f'Datasets matrix representations imported')

# reading the raw data with pandas

# 1st dataset
Xtr0 = pd.read_csv("data/Xtr0.csv")
Xtr0=Xtr0['seq'].tolist()

Xte0 = pd.read_csv("data/Xte0.csv")
Xte0=Xte0['seq'].tolist()

# 2nd dataset
Xtr1 = pd.read_csv("data/Xtr1.csv")
Xtr1=Xtr1['seq'].tolist()

Xte1 = pd.read_csv("data/Xte1.csv")
Xte1=Xte1['seq'].tolist()

# 3rd dataset
Xtr2 = pd.read_csv("data/Xtr2.csv")
Xtr2=Xtr2['seq'].tolist()

Xte2 = pd.read_csv("data/Xte2.csv")
Xte2=Xte2['seq'].tolist()

print(f'Datasets + their matrix representations imported')

"""# 2. Helper Functions"""

def export_predict(pred_1,pred_2,pred_3,file_name):
    preds = pd.DataFrame({'Id':np.array(range(3000)),'Bound':np.concatenate([pred_1,pred_2,pred_3])})
    preds.to_csv(file_name,index=False)


def ensure_2D(array):

    # If input is scalar raise error
    if array.ndim == 0:
        raise ValueError(
            "Expected 2D array, got scalar array instead:\narray={}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample.".format(array))
    # If input is 1D raise error
    if array.ndim == 1:
        raise ValueError(
            "Expected 2D array, got 1D array instead:\narray={}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample.".format(array))
        
def check_y(X, y):

    y = y.reshape(-1)
    if len(y) != len(X):
      raise ValueError(
            "Expected arrayof shape ({},) or ({},1). Got instead:\narray={}."
            .format(len(X), len(X), array))

def train_test_split(X, y, train_frac=0.75):
    
    shuffle = np.random.permutation(len(X))
    last_train = int(train_frac*len(X))
    
    X_rnd = X[shuffle]
    y_rnd = y[shuffle]

    return X_rnd[:last_train], y_rnd[:last_train], X_rnd[last_train:], y_rnd[last_train:]


def train_test_split_on_gram(K, y, train_frac=0.75):

    shuffle = np.random.permutation(len(K))
    last_train = int(train_frac*len(K))
    idxs_train = shuffle[:last_train]
    idxs_test = shuffle[last_train:]

    K_train = K[idxs_train].T[idxs_train].T
    K_test_train = K[idxs_test].T[idxs_train].T

    return K_train, y[idxs_train], K_test_train, y[idxs_test]



def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)



def remap_zero_minus_one(y):
    new_y = y.copy()
    new_y[y==0] = -1
    return new_y

def remap_minus_one_zero(y):
    new_y = y.copy()
    new_y[y==-1] = 0
    return new_y



"""# 3. Ridge Regression"""

#Ridge without kernel
def ridgeEstimator(X,y,lambd):
    """
    Params:
    ######
    X      is a (n_samples, n_features) array : the training inputs
    y      is a (n_samples,) array : the training labels
    lambd  is a postive scalar (regularization term)

    Returns:
    ######
    theta : a (n_features,) array. The parameters of the linear model estimation
    """
    
    # check size of X
    ensure_2D(X)
    # check size of y
    check_y(X,y)

    # the labels must be {-1,+1}
    y = remap_zero_minus_one(y)
    
    (n,d) = X.shape
    # decide to invert a (n,n) or a (d,d) matrix
    if d <= n: # standard ridge regression
    
        # compute the ridge estimator
        cov = X.T @ X
        reg = lambd * n * np.eye(d)

        theta = linalg.solve( cov+reg , X.T @ y ) # résout le système linéaire
    
    else: # kernel ridge regression with a linear kernel

        gram = X @ X.T
        reg = lambd * n * np.eye(n)

        theta = X.T @ linalg.solve( gram+reg , y ) # résout le système linéaire

    return theta


def predictLinear(X, theta):

    # check size of X
    ensure_2D(X)
    # linear prediction
    preds = np.sign(X @ theta).astype(int)
    # remap {-1,1} to {0,1}
    preds = remap_minus_one_zero(preds)

    return preds




"""# 4. KERNEL Ridge Regression"""

### Kernels on the matrix representation


### GAUSSIAN KERNEL
def gaussian_kernel(x, y, gamma=1):
    return np.exp(-gamma * np.sum((x - y)**2))

### LAPLACIAN KERNEL
def laplacian_kernel(x, y, gamma=1):
    return np.exp(-gamma * np.sum(abs(x - y)))

### POLYNOMIAL KERNEL
def ploynomial_kernel(x, y, d=3, gamma=1, c=1):
    return (gamma * x@y + c)**d



def compute_gram_from_kernel(X1,X2, kernel_func, params):
    n1 = len(X1)
    n2 = len(X2)
    K = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            K[i][j] = kernel_func(X1[i],X2[j], params)
    return K


#######################################
### KRR 
#######################################

def kernelRidgeEstimator(K,y,lambd):
    """
    Params:
    ######
    K : (n_samples, n_samples) array. Kernel matrix of the input training data
    y : (n_samples,) array. Training labels
    lambd : positive scalar. Regularization parameter
    
    Returns:
    ######
    alpha : (n_samples,) array. The weights to give to each training data point.
    """
    n = len(K)

    # the labels must be {-1,+1}
    y = remap_zero_minus_one(y)

    reg = lambd * n * np.eye(n)
    
    alpha = linalg.solve( K + reg , y ) # solve the linear system
    
    return alpha

def predict_from_gram(K_test_train, alpha):

    preds = np.sign(K_test_train @ alpha).astype(int)
    
    # remap {-1,1} to {0,1}
    preds = remap_minus_one_zero(preds)
  
    return preds



######################################################
### SPECTRUM KERNEL
######################################################

def spectrum_kernel(x1,x2,k=6, normalize=False):
    
    # initialize the dictionaries of subsequences
    phi1=dict()
    phi2=dict()
    
    for i in range(len(x1)-k+1):
        
        # extract a subsequence of x1
        subsequence = x1[i:k+i]
        
        # add the subsequence found to the dictionary 1
        if subsequence not in phi1:
            phi1[subsequence] = 1
        else:
            phi1[subsequence] += 1
        
    for i in range(len(x2)-k+1):

        # extract a subsequence of x2
        subsequence = x2[i:k+i]
        
        # add the subsequence found to the dictionary 2
        if subsequence not in phi2:
            phi2[subsequence]=1
        else:
            phi2[subsequence]+=1
    
    # compute the scalar product between the two representations obtained
    scalar_prod = 0
    for subsequence in phi1.keys() & phi2.keys():
        scalar_prod += phi1[subsequence] * phi2[subsequence]
    
    if normalize:
      norm_phi1 = ( sum(e**2 for e in phi1.values()) )**0.5
      norm_phi2 = ( sum(e**2 for e in phi2.values()) )**0.5

      scalar_prod /= (norm_phi1 * norm_phi2)
    
    return scalar_product



#########################################################
### FASTER IMPLEMENTATION TO COMPUTE THE GRAM MATRIX
#########################################################

def compute_spectrum_kernel_Gram(X,k=6, normalize=False):
  """
  implement the function above but with ina  slightly more efficient version

  In X, each row represents a sequence ('ATCGCTTGA...)
  """

    # initialize the dictionaries of subsequences
  phis = []
  for sequence_id in range(len(X)):

    phi = dict()
    
    for i in range(len(X[sequence_id])-k+1):
        
      # extract a subsequence
      subsequence = X[sequence_id][i:k+i]
      
      # add the subsequence found to its dictionary 
      if subsequence not in phi:
        phi[subsequence] = 1
      else:
        phi[subsequence] += 1

    phis.append(phi)

  norms_phis = np.array([(sum(e**2 for e in phis[i].values()))**0.5 for i in range(len(phis))]).reshape(1, -1)
  
  Gram = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    for j in range(i+1,len(X)):

      # compute the scalar products between the representations obtained
      scalar_prod = 0
      for subsequence in phis[i].keys() & phis[j].keys():
        scalar_prod += phis[i][subsequence] * phis[j][subsequence]
      
      Gram[i,j] = scalar_prod
      Gram[j,i] = scalar_prod
  
  if normalize:
    Gram = Gram / np.repeat(norms_phis, len(X), axis=0)
    Gram = Gram / np.repeat(norms_phis, len(X), axis=0).T
    Gram = Gram + np.eye(len(X))
  else:
    Gram = Gram + np.diag(norms_phis.reshape(-1))
  
  return Gram



#######################################################################
### COMPUTE THE SUM OF GRAM MATRICES USING SUMS OF SPECTRUM KERNELS
#######################################################################

def compute_sum_spectrum_kernels_Gram(X, list_k, normalize=False):
  """
  Allows to use the spectrum kernel for multiple subsequences-lenghts and sum the kernels
  """

  Gram = np.zeros((len(X),len(X)))
  for k in list_k:
    Gram = Gram + compute_spectrum_kernel_Gram(X,k,normalize)

  return Gram / len(list_k)



#######################################################################
### MAIN SCRIPT
#######################################################################

def start():

	# predictions for dataset 0

	k_0 = [6,7,8]
	lambd_0 = 0.001

	X_te_plus_tr_0 = Xte0 + Xtr0

	K_te_plus_tr_0 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_0, k_0, normalize=True)


	# center the Gram matrix
	n = len(K_te_plus_tr_0)
	I = np.eye(n)
	U = np.ones((n,n)) / n
	K_te_plus_tr_0 = (I-U) @ K_te_plus_tr_0 @ (I-U)

	K_tr_0 = K_te_plus_tr_0[1000:,1000:]
	K_te_tr_0 = K_te_plus_tr_0[:1000,1000:]


	#KRR predictions:
	alpha_0 = kernelRidgeEstimator(K_tr_0, train_labels_0, lambd=lambd_0)

	y_pred_0 = predict_from_gram(K_te_tr_0, alpha_0)
	print(f'Predictions for dataset 0 done...')

	# predictions for dataset 1

	k_1 = [5,7,8]
	lambd_1 = 0.001

	X_te_plus_tr_1 = Xte1 + Xtr1

	K_te_plus_tr_1 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_1, k_1, normalize=True)


	# center the Gram matrix
	n = len(K_te_plus_tr_1)
	I = np.eye(n)
	U = np.ones((n,n)) / n
	K_te_plus_tr_1 = (I-U) @ K_te_plus_tr_1 @ (I-U)

	K_tr_1 = K_te_plus_tr_1[1000:,1000:]
	K_te_tr_1 = K_te_plus_tr_1[:1000,1000:]


	#KRR predictions:
	alpha_1 = kernelRidgeEstimator(K_tr_1, train_labels_1, lambd=lambd_1)

	y_pred_1 = predict_from_gram(K_te_tr_1, alpha_1)
	print(f'Predictions for dataset 1 done...')

	# predictions for dataset 2

	k_2 = [6,8]
	lambd_2 = 0.0003

	X_te_plus_tr_2 = Xte2 + Xtr2

	K_te_plus_tr_2 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_2, k_2, normalize=True)


	# center the Gram matrix
	n = len(K_te_plus_tr_2)
	I = np.eye(n)
	U = np.ones((n,n)) / n
	K_te_plus_tr_2 = (I-U) @ K_te_plus_tr_2 @ (I-U)

	K_tr_2 = K_te_plus_tr_2[1000:,1000:]
	K_te_tr_2 = K_te_plus_tr_2[:1000,1000:]


	#KRR predictions:
	alpha_2 = kernelRidgeEstimator(K_tr_2, train_labels_2, lambd=lambd_2)

	y_pred_2 = predict_from_gram(K_te_tr_2, alpha_2)

	
	export_predict(y_pred_0,y_pred_1,y_pred_2,'Yte.csv')
	print(f'Predictions exported in Yte.csv')


if __name__ == '__main__':

	start()


"""Ensemble method : learn a KR estimator on 15 random subsets of the trainingset of size 1500 and vote.
Gives a slightly lower result than without voting."""

# predictions for dataset 0

# k_0 = [6,7,8]
# lambd_0 = 0.001


# X_te_plus_tr_0 = Xte0 + Xtr0

# K_te_plus_tr_0 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_0, k_0, normalize=True)

# y_pred_0 = []

# for n_model in range(15):

#   shuffle = np.random.permutation(2000)
#   last_train = int(0.75*2000)
#   idxs_train = shuffle[:last_train]
#   idxs = np.concatenate([np.arange(1000), 1000+idxs_train])

#   K_te_plus_part_tr_0 = K_te_plus_tr_0[idxs].T[idxs].T
#   part_train_labels_0 = train_labels_0[idxs_train]

#   # center the Gram matrix
#   n = len(K_te_plus_part_tr_0)
#   I = np.eye(n)
#   U = np.ones((n,n)) / n
#   K_te_plus_part_tr_0 = (I-U) @ K_te_plus_part_tr_0 @ (I-U)

#   K_part_tr_0 = K_te_plus_part_tr_0[1000:,1000:]
#   K_te_part_tr_0 = K_te_plus_part_tr_0[:1000,1000:]


#   #KRR predictions:
#   alpha_0 = kernelRidgeEstimator(K_part_tr_0, part_train_labels_0, lambd=lambd_0)

#   y_pred_0.append(predict_from_gram(K_te_part_tr_0, alpha_0))

# y_pred_0 = np.array(y_pred_0)
# y_pred_merged_0 = np.round(y_pred_0.mean(axis=0)).astype(int)

# # predictions for dataset 1

# k_1 = [5,7,8]
# lambd_1 = 0.001


# X_te_plus_tr_1 = Xte1 + Xtr1

# K_te_plus_tr_1 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_1, k_1, normalize=True)

# y_pred_1 = []

# for n_model in range(15):

#   shuffle = np.random.permutation(2000)
#   last_train = int(0.75*2000)
#   idxs_train = shuffle[:last_train]
#   idxs = np.concatenate([np.arange(1000), 1000+idxs_train])

#   K_te_plus_part_tr_1 = K_te_plus_tr_1[idxs].T[idxs].T
#   part_train_labels_1 = train_labels_1[idxs_train]

#   # center the Gram matrix
#   n = len(K_te_plus_part_tr_1)
#   I = np.eye(n)
#   U = np.ones((n,n)) / n
#   K_te_plus_part_tr_1 = (I-U) @ K_te_plus_part_tr_1 @ (I-U)

#   K_part_tr_1 = K_te_plus_part_tr_1[1000:,1000:]
#   K_te_part_tr_1 = K_te_plus_part_tr_1[:1000,1000:]


#   #KRR predictions:
#   alpha_1 = kernelRidgeEstimator(K_part_tr_1, part_train_labels_1, lambd=lambd_1)

#   y_pred_1.append(predict_from_gram(K_te_part_tr_1, alpha_1))

# y_pred_1 = np.array(y_pred_1)
# y_pred_merged_1 = np.round(y_pred_1.mean(axis=0)).astype(int)

# # predictions for dataset 2

# k_2 = [6,8]
# lambd_2 = 0.0003


# X_te_plus_tr_2 = Xte2 + Xtr2

# K_te_plus_tr_2 = compute_sum_spectrum_kernels_Gram(X_te_plus_tr_2, k_2, normalize=True)

# y_pred_2 = []

# for n_model in range(15):

#   shuffle = np.random.permutation(2000)
#   last_train = int(0.75*2000)
#   idxs_train = shuffle[:last_train]
#   idxs = np.concatenate([np.arange(1000), 1000+idxs_train])

#   K_te_plus_part_tr_2 = K_te_plus_tr_2[idxs].T[idxs].T
#   part_train_labels_2 = train_labels_2[idxs_train]

#   # center the Gram matrix
#   n = len(K_te_plus_part_tr_2)
#   I = np.eye(n)
#   U = np.ones((n,n)) / n
#   K_te_plus_part_tr_2 = (I-U) @ K_te_plus_part_tr_2 @ (I-U)

#   K_part_tr_2 = K_te_plus_part_tr_2[1000:,1000:]
#   K_te_part_tr_2 = K_te_plus_part_tr_2[:1000,1000:]


#   #KRR predictions:
#   alpha_2 = kernelRidgeEstimator(K_part_tr_2, part_train_labels_2, lambd=lambd_2)

#   y_pred_2.append(predict_from_gram(K_te_part_tr_2, alpha_2))

# y_pred_2 = np.array(y_pred_2)
# y_pred_merged_2 = np.round(y_pred_2.mean(axis=0)).astype(int)

# export_predict(y_pred_merged_0,y_pred_merged_1,y_pred_merged_2,'Yte_vote.csv')


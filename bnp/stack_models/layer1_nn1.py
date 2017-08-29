# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:03:06 2015

@author: branden

Note to self:
If you get error:
ERROR (theano.sandbox.cuda): Failed to compile cuda_ndarray.cu: libcublas.so.7.5: cannot open shared object file: No such file or directory

when starting theano try "sudo ldconfig /usr/local/cuda-7.5/lib64" in terminal
"""
import numpy as np
import pandas as pd
import gc

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import GaussianNoiseLayer
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import rectify
from lasagne.objectives import     binary_crossentropy
from lasagne.init import Constant, GlorotUniform
from lasagne.updates import nesterov_momentum, adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

### START WITH DATA TRANSFORMATION CREATED IN R
ts1Trans = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/ts2Trans_v8.csv") 
cvFolds = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/cvFolds.csv")

t1nn = ts1Trans.loc[ts1Trans['filter']==0,]
s1nn = ts1Trans.loc[ts1Trans['filter']==2,]

labels = ts1Trans.loc[ts1Trans['filter']==0,'target']
t1nn_id = ts1Trans.loc[ts1Trans['filter']==0,'ID']
encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.float32)




del ts1Trans
gc.collect()



scaler = StandardScaler()
#index = range(3,5469) + range(6832,6839)
t1nn_feats = t1nn.iloc[:,3:]
t1nn_feats = np.log(np.maximum(t1nn_feats,0) + 1)
t1nn_feats = scaler.fit_transform(t1nn_feats).astype(np.float32)
s1nn_feats = s1nn.iloc[:,3:]
s1nn_feats = np.log(np.maximum(s1nn_feats,0) + 1)
s1nn_feats = scaler.transform(s1nn_feats).astype(np.float32)
#
#index1 = range(3,115) + range(6832,6839)
pca = PCA(n_components= .999, whiten=True)
t1nn_pca = pca.fit_transform(t1nn_feats).astype(np.float32)
s1nn_pca = pca.transform(s1nn_feats).astype(np.float32)
#
#pca1 = PCA(n_components= 700, whiten=True)
#t1nn_pca1 = pca1.fit_transform(t1nn_feats[:,115:5469]).astype(np.float32)
#s1nn_pca1 = pca1.transform(s1nn_feats[:,115:5469]).astype(np.float32)
#
#pca2 = PCA(n_components= 200, whiten=True)
#t1nn_pca2 = pca2.fit_transform(t1nn_feats[:,5469:6832]).astype(np.float32)
#s1nn_pca2 = pca2.transform(s1nn_feats[:,5469:6832]).astype(np.float32)

t1nn_conc = np.concatenate([t1nn_pca, t1nn_feats], axis=1)
#t1nn_conc = np.concatenate([t1nn_pca, t1nn_pca1], axis=1)
s1nn_conc = np.concatenate([s1nn_pca, s1nn_feats], axis=1)

#t1nn_conc_df = pd.DataFrame(t1nn_conc)
#s1nn_conc_df = pd.DataFrame(s1nn_conc)
#
#t1nn_conc_df.to_csv("/home/branden/Documents/kaggle/walmart/data_trans/t1nn_conc_simil.csv", index=False)
#s1nn_conc_df.to_csv("/home/branden/Documents/kaggle/walmart/data_trans/s1nn_conc_simil.csv", index=False)
#
##
##
##t1nn_comb = np.concatenate([t1nn_conc, t1_dept_simil], axis=1).astype(np.float32)
##s1nn_comb = np.concatenate([s1nn_conc, s1_dept_simil], axis=1).astype(np.float32)
##

num_classes = len(encoder.classes_)
num_features = t1nn_conc.shape[1]


import theano
t1nn_conc_shared = theano.shared(t1nn_conc, borrow=True)

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

#class AdjustVariableRate(object):
#    def __init__(self, name, start=0.03, stop=.001):
#        self.name = name
#        self.start, self.stop = start, stop
#        self.ls = None
#
#    def __call__(self, nn, train_history):
#        if self.ls is None:
#            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
#
#        epoch = train_history[-1]['epoch']
#        new_value = float32(self.ls[epoch - 1])
#        getattr(nn, self.name).set_value(new_value)

from nolearn.lasagne import BatchIterator

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

layers0 = [('input', InputLayer),
           ('inputDropout0', DropoutLayer),          
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
  #         ('noise0', GaussianNoiseLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
#           ('dense2', DenseLayer),
#           ('dropout2', DropoutLayer),
           ('output', DenseLayer)]



#0.686160
# inDrop=0.2, den0=1000, den0drop=.6, den1=1000, den1drop=0.6
from theano import tensor as T

np.random.seed(5)        
net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.5,
                 dense0_num_units=80,
                 dense0_W=GlorotUniform(),
                 dense0_b = Constant(1.0),
                 dense0_nonlinearity=rectify,
                 dropout0_p=0.2,
#                 noise0_sigma=2,
                 dense1_num_units=80,
                 dense1_W=GlorotUniform(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.2,
#                 dense2_num_units=50,
#                 dense2_W=GlorotUniform(),
#                 dense2_nonlinearity=rectify,
#                 dense2_b = Constant(1.0),
#                 dropout2_p=0.2,
                 output_num_units=1,
                 output_nonlinearity=sigmoid,
                 objective_loss_function=binary_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0001), borrow=True),
#                 update_momentum=theano.shared(float32(0.001), borrow=True),
                 update_beta1=0.9,
                 update_beta2=0.99,
                 update_epsilon=1e-06,
                 on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.3, stop=0.05),
#                     AdjustVariable('update_momentum', start=0.001, stop=0.00299),
#                     EarlyStopping(patience=200),
                     ],               
                 regression=True,
                 train_split=TrainSplit(eval_size=0.0),
                 y_tensor_type = T.matrix,
                 verbose=1,
                 batch_iterator_train=BatchIterator(3200),
                 max_epochs=230)

#np.random.seed(7)
#net0_clone = clone(net0)
#net0_clone.fit(t1nn_conc_shared.get_value(), y)
#net0_clone.fit(X_encoded_shared.get_value(), y)

cv_by_hand = [(np.where(cvFolds != fold)[0], np.where(cvFolds == fold)[0])
               for fold in np.unique(cvFolds)]


foldPred = np.zeros((t1nn_conc_shared.get_value().shape[0], 1))
bags = 10
for iter in xrange(0,bags):
        for fold in xrange(0,np.max(cvFolds)):
            np.random.seed(iter + 56)
            net0_clone = clone(net0)
            net0_clone.fit(t1nn_conc_shared.get_value()[cv_by_hand[fold][0],:], y[cv_by_hand[fold][0]])
            foldPred[cv_by_hand[fold][1],:] += net0_clone.predict_proba(t1nn_conc_shared.get_value()[cv_by_hand[fold][1],:])
foldPred = foldPred/bags            

# Load sample submission
samp = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/sample_submission.csv") 
#Convert CV preds array to DataFrame with column names
foldPred = pd.DataFrame(foldPred, columns=samp.columns[1:])
# Add VisitNumber column
foldPred.insert(0, 'ID', t1nn_id)
# Save 
foldPred.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_nn1.csv", index=False)


np.random.seed(7)        
net1 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.5,
                 dense0_num_units=80,
                 dense0_W=GlorotUniform(),
                 dense0_b = Constant(1.0),
                 dense0_nonlinearity=rectify,
                 dropout0_p=0.2,
#                 noise0_sigma=2,
                 dense1_num_units=80,
                 dense1_W=GlorotUniform(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.2,
#                 dense2_num_units=50,
#                 dense2_W=GlorotUniform(),
#                 dense2_nonlinearity=rectify,
#                 dense2_b = Constant(1.0),
#                 dropout2_p=0.2,
                 output_num_units=1,
                 output_nonlinearity=sigmoid,
                 objective_loss_function=binary_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0001), borrow=True),
#                 update_momentum=theano.shared(float32(0.001), borrow=True),
                 update_beta1=0.9,
                 update_beta2=0.99,
                 update_epsilon=1e-06,
                 on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.3, stop=0.05),
#                     AdjustVariable('update_momentum', start=0.001, stop=0.00299),
#                     EarlyStopping(patience=200),
                     ],               
                 regression=True,
                 train_split=TrainSplit(eval_size=0.00),
                 y_tensor_type = T.matrix,
                 verbose=1,
                 batch_iterator_train=BatchIterator(3200),
                 max_epochs=260)




test_pred= np.zeros((s1nn_pca.shape[0], 1))
for iter in xrange(0,bags):
    np.random.seed(iter + 59)
    net1_clone = clone(net1)
    net1_clone.fit(t1nn_conc_shared.get_value(), y)
    test_pred += net1_clone.predict_proba(s1nn_conc)
test_pred_adj = test_pred/bags    


  
samp = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/sample_submission.csv") 
#Convert CV preds array to DataFrame with column names
test_pred_frame = pd.DataFrame(test_pred_adj, columns=samp.columns[1:])
# Add VisitNumber column
s1nn = s1nn.reset_index(drop=True)
test_pred_frame.insert(0, 'ID', s1nn['ID'])
# Save 
test_pred_frame.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_nn1.csv", index=False)











# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:03:06 2015

@author: branden
"""

import numpy as np
import pandas as pd
import scipy as sp
import gc

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone

from lasagne.layers import get_output, DenseLayer, InputLayer, DropoutLayer,GaussianNoiseLayer
from lasagne.nonlinearities import softmax, sigmoid, rectify, identity
from lasagne.objectives import binary_crossentropy
from lasagne.init import Constant, GlorotUniform, GlorotNormal
from lasagne.updates import nesterov_momentum, adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

# Import CV folds for layer 2
cvFolds = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/cvFolds_lay2.csv")

##############################
### CV PREDICTIONS
##############################
xgb18cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb18.csv")
xgb20cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb20.csv")
xgb21cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb21.csv")
xgb22cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb22.csv")
xgb24cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb24.csv")
xgb25cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb25.csv")

nn3cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_nn3.csv")

et1cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_et1.csv")
et2cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_et2.csv")

glmnet1cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_glmnet1.csv")
glmnet2cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_glmnet2.csv")
glmnet3cv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_glmnet3.csv")

rafacv = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/Rafa/branden_exp/probs_stage_1/train.csv")
rafacv.columns.values[0]="ID"
rafacv = rafacv.reset_index().sort_values('ID')



train_ID = xgb18cv['ID']

xgb18cv.drop("ID", axis=1, inplace=True)
xgb20cv.drop("ID", axis=1, inplace=True)
xgb21cv.drop("ID", axis=1, inplace=True)
xgb22cv.drop("ID", axis=1, inplace=True)
xgb24cv.drop("ID", axis=1, inplace=True)
xgb25cv.drop("ID", axis=1, inplace=True)

nn3cv.drop("ID", axis=1, inplace=True)

et1cv.drop("ID", axis=1, inplace=True)
et2cv.drop("ID", axis=1, inplace=True)

glmnet1cv.drop("ID", axis=1, inplace=True)
glmnet2cv.drop("ID", axis=1, inplace=True)
glmnet3cv.drop("ID", axis=1, inplace=True)

rafacv.drop("ID", axis=1, inplace=True)

lay1preds = np.concatenate([xgb18cv,xgb20cv,xgb21cv,xgb22cv,xgb24cv, xgb25cv, nn3cv, et1cv, et2cv, glmnet1cv, glmnet2cv, glmnet3cv, rafacv], axis=1)

train = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/train.csv")
train = train.iloc[:,0:2]

labels = train['target']
encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.float32)


##############################
### TEST PREDICTIONS
##############################
xgb18test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb18.csv")
xgb20test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb20.csv")
xgb21test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb21.csv")
xgb22test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb22.csv")
xgb24test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb24.csv")
xgb25test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb25.csv")

nn3test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_nn3.csv")

et1test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_et1.csv")
et2test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_et2.csv")

glmnet1test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_glmnet1.csv")
glmnet2test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_glmnet2.csv")
glmnet3test = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_glmnet3.csv")

rafatest = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/Rafa/branden_exp/probs_stage_1/test.csv")
rafatest.columns.values[0]="ID"
rafatest = rafatest.reset_index().sort_values('ID')



train_ID = xgb18test['ID']

xgb18test.drop("ID", axis=1, inplace=True)
xgb20test.drop("ID", axis=1, inplace=True)
xgb21test.drop("ID", axis=1, inplace=True)
xgb22test.drop("ID", axis=1, inplace=True)
xgb24test.drop("ID", axis=1, inplace=True)
xgb25test.drop("ID", axis=1, inplace=True)

nn3test.drop("ID", axis=1, inplace=True)

et1test.drop("ID", axis=1, inplace=True)
et2test.drop("ID", axis=1, inplace=True)

glmnet1test.drop("ID", axis=1, inplace=True)
glmnet2test.drop("ID", axis=1, inplace=True)
glmnet3test.drop("ID", axis=1, inplace=True)

rafatest.drop("ID", axis=1, inplace=True)

lay1testpreds = np.concatenate([xgb18test,xgb20test,xgb21test,xgb22test,xgb24test, xgb25test, nn3test, et1test, et2test, glmnet1test, glmnet2test, glmnet3test, rafatest], axis=1)



gc.collect()


scaler = StandardScaler()
cv_feats = sp.maximum(sp.minimum(lay1preds, 1-1e-15), 1e-15)
cv_feats = np.log(cv_feats/(1-cv_feats))
cv_feats = scaler.fit_transform(cv_feats).astype(np.float32)
test_feats = sp.maximum(sp.minimum(lay1testpreds, 1-1e-15), 1e-15)
test_feats = np.log(test_feats/(1-test_feats))
test_feats = scaler.transform(test_feats).astype(np.float32)



num_classes = len(encoder.classes_)
num_features = cv_feats.shape[1]


import theano
lay1preds_shared = theano.shared(cv_feats, borrow=True)

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

np.random.seed(8)        
net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.0,
                 dense0_num_units=128,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dense0_nonlinearity=rectify,
                 dropout0_p=0.1,
#                 noise0_sigma=2,
                 dense1_num_units=128,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.1,
#                 dense2_num_units=64,
#                 dense2_W=GlorotUniform(),
#                 dense2_nonlinearity=rectify,
#                 dense2_b = Constant(1.0),
#                 dropout2_p=0.4,
                 output_num_units=1,
                 output_nonlinearity=sigmoid,
                 objective_loss_function=binary_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0003), borrow=True),
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
                 batch_iterator_train=BatchIterator(800),
                 max_epochs=100)

#np.random.seed(8)
#net0_clone = clone(net0)
#net0_clone.fit(lay1preds_shared.get_value(), y)

cv_by_hand = [(np.where(cvFolds != fold)[0], np.where(cvFolds == fold)[0])
               for fold in np.unique(cvFolds)]


foldPred = np.zeros((lay1preds_shared.get_value().shape[0], 1))
bags = 20
for iter in xrange(0,bags):
        for fold in xrange(0,np.max(cvFolds)):
            print("iter ", iter," fold ", fold )
            np.random.seed(iter + 56)
            net0_clone = clone(net0)
            net0_clone.fit(lay1preds_shared.get_value()[cv_by_hand[fold][0],:], y[cv_by_hand[fold][0]])
            foldPred[cv_by_hand[fold][1],:] += net0_clone.predict_proba(lay1preds_shared.get_value()[cv_by_hand[fold][1],:])
foldPredAdj = foldPred/bags            

        



# Load sample submission
samp = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/sample_submission.csv") 
#Convert CV preds array to DataFrame with column names
foldPredFrame = pd.DataFrame(foldPredAdj, columns=samp.columns[1:])
# Add ID column
foldPredFrame.insert(0, 'ID', train_ID)
# Save 
foldPredFrame.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/layer2Preds/cvPreds_lay2_nn1.csv", index=False)



np.random.seed(5)        
net1 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.0,
                 dense0_num_units=128,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dense0_nonlinearity=rectify,
                 dropout0_p=0.1,
#                 noise0_sigma=2,
                 dense1_num_units=128,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.1,
#                 dense2_num_units=64,
#                 dense2_W=GlorotUniform(),
#                 dense2_nonlinearity=rectify,
#                 dense2_b = Constant(1.0),
#                 dropout2_p=0.4,
                 output_num_units=1,
                 output_nonlinearity=sigmoid,
                 objective_loss_function=binary_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0003), borrow=True),
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
                 batch_iterator_train=BatchIterator(800),
                 max_epochs=120)


test_pred = np.zeros((test_feats.shape[0], 1))
for iter in xrange(0,bags):
    print(iter)
    np.random.seed(iter + 59)
    net1_clone = clone(net1)
    net1_clone.fit(lay1preds_shared.get_value(), y)
    test_pred += net1_clone.predict_proba(test_feats)
test_pred_adj = test_pred/bags    
    
samp = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/sample_submission.csv") 
#Convert CV preds array to DataFrame with column names
test_pred_frame = pd.DataFrame(test_pred_adj, columns=samp.columns[1:])
# Add ID column
test_pred_frame.insert(0, 'ID', samp['ID'])
# Save 
test_pred_frame.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/layer2Preds/testPreds_lay2_nn1.csv", index=False)


















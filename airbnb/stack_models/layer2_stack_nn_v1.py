# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:03:06 2015

@author: branden
"""

import numpy as np
import pandas as pd
import gc

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone

from lasagne.layers import get_output, DenseLayer, InputLayer, DropoutLayer,GaussianNoiseLayer
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.nonlinearities import softmax, sigmoid, rectify, identity
from lasagne.updates import adam
from lasagne.layers import get_all_params
from lasagne.objectives import categorical_crossentropy
from lasagne.init import Constant, GlorotUniform, GlorotNormal
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

# Import CV folds for layer 2
cvFolds = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/cvFolds_lay2_k6.csv")

### START WITH DATA TRANSFORMATION CREATED IN R
xgb1cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb1.csv")
xgb2cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb2.csv")
xgb3cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb3.csv")
xgb4cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb4.csv")
xgb5cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb5.csv")
xgb6cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb6.csv")
xgb7cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb7.csv")
xgb8cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb8.csv")
xgb9cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb9.csv")
xgb10cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb10.csv")
xgb11cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb11.csv")
xgb12cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb12.csv")
xgb13cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb13.csv")
xgb14cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_xgb14.csv")
nn1cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_nn1.csv")
nn2cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_nn2.csv")
nn3cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_nn3.csv")
nn4cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_nn4.csv")
rf1cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_rf1.csv")
glmnet1cv = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_glmnet1.csv")

train_id = xgb11cv['id']

xgb1cv.drop("id", axis=1, inplace=True)
xgb2cv.drop("id", axis=1, inplace=True)
xgb3cv.drop("id", axis=1, inplace=True)
xgb4cv.drop("id", axis=1, inplace=True)
xgb5cv.drop("id", axis=1, inplace=True)
xgb6cv.drop("id", axis=1, inplace=True)
xgb7cv.drop("id", axis=1, inplace=True)
xgb8cv.drop("id", axis=1, inplace=True)
xgb9cv.drop("id", axis=1, inplace=True)
xgb10cv.drop("id", axis=1, inplace=True)
xgb11cv.drop("id", axis=1, inplace=True)
xgb12cv.drop("id", axis=1, inplace=True)
xgb13cv.drop("id", axis=1, inplace=True)
xgb14cv.drop("id", axis=1, inplace=True)
nn1cv.drop("id", axis=1, inplace=True)
nn2cv.drop("id", axis=1, inplace=True)
nn3cv.drop("id", axis=1, inplace=True)
nn4cv.drop("id", axis=1, inplace=True)
rf1cv.drop("id", axis=1, inplace=True)
glmnet1cv.drop("id", axis=1, inplace=True)
#glmnet1cv.fillna(0, inplace=True)

lay1preds = np.concatenate([xgb1cv, xgb2cv, xgb6cv,xgb7cv,xgb8cv,xgb9cv,xgb10cv, xgb11cv, xgb12cv, xgb13cv, xgb14cv, nn1cv, nn2cv, nn3cv, nn4cv, rf1cv,glmnet1cv], axis=1).astype(np.float32)

train = pd.read_csv("/home/branden/Documents/kaggle/airbnb/train_users_2.csv")
train = train.sort('id')

countryDestinations = pd.unique(train.country_destination.ravel())
countryDestinations = sorted(countryDestinations, key=str.lower)
classes = np.array(range(12))
countryClasses = pd.DataFrame({"country_destination": countryDestinations, "class": classes})
train = train.merge(countryClasses, how="left", on="country_destination")


labels = train['class']
encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.int32)


gc.collect()


#scaler = StandardScaler()
##index = range(3,5469) + range(6832,6839)
#t1nn_feats = t1nn
#t1nn_feats = np.log(np.maximum(t1nn_feats,0) + 1)
#t1nn_feats = scaler.fit_transform(t1nn_feats).astype(np.float32)
#s1nn_feats = s1nn
#s1nn_feats = np.log(np.maximum(s1nn_feats,0) + 1)
#s1nn_feats = scaler.transform(s1nn_feats).astype(np.float32)
#
#index1 = range(3,116) + range(6833,6839)
#pca = PCA(n_components= .999, whiten=True)
#t1nn_pca = pca.fit_transform(t1nn_feats[:,index1]).astype(np.float32)
#s1nn_pca = pca.transform(s1nn_feats[:,index1]).astype(np.float32)
#
#pca1 = PCA(n_components= 1200, whiten=True)
#t1nn_pca1 = pca1.fit_transform(t1nn_feats[:,116:5468]).astype(np.float32)
#s1nn_pca1 = pca1.transform(s1nn_feats[:,116:5468]).astype(np.float32)
#
#pca2 = PCA(n_components= 200, whiten=True)
#t1nn_pca2 = pca2.fit_transform(t1nn_feats[:,5470:6832]).astype(np.float32)
#s1nn_pca2 = pca2.transform(s1nn_feats[:,5470:6832]).astype(np.float32)
#
#t1nn_conc = np.concatenate([t1nn_pca, t1nn_feats[:,index1], t1nn_pca1, t1nn_pca2], axis=1)
##t1nn_conc = np.concatenate([t1nn_pca, t1nn_pca1], axis=1)
#s1nn_conc = np.concatenate([s1nn_pca, s1nn_feats[:,index1], s1nn_pca1, s1nn_pca2], axis=1)




#del t1nn, s1nn
#gc.collect()


num_classes = len(encoder.classes_)
num_features = lay1preds.shape[1]


import theano
lay1preds_shared = theano.shared(lay1preds, borrow=True)

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

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - theano.tensor.log(theano.tensor.sum(theano.tensor.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -theano.tensor.sum(log_predictions[targets] * log_predictions, axis=1)

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



def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses




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



# 1.03300
# inDrop=0.6, den0=64, den0drop=0.2, den1=64, den1drop=0.2

np.random.seed(16)        
net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.0,
                 dense0_num_units=128,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dropout0_p=0.05,
                 dense0_nonlinearity=rectify,
#                 noise0_sigma=2,
                 dense1_num_units=128,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.05,
#                 dense2_num_units=64,
#                 dense2_W=GlorotUniform(),
#                 dense2_b = Constant(1.0),
#                 dense2_nonlinearity=rectify,
#                 dropout2_p=0.05,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 objective=regularization_objective,
#                 objective_lambda1=0.00000001,
#                 objective_loss_function=categorical_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0001), borrow=True),
#                 update_momentum=theano.shared(float32(0.001), borrow=True),
                 update_beta1=0.9,
                 update_beta2=0.99,
                 update_epsilon=1e-06,
                 on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.3, stop=0.05),
#                     AdjustVariable('update_momentum', start=0.001, stop=0.00299),
#                     EarlyStopping(patience=100),
                     ],               
                 
                 train_split=TrainSplit(eval_size=0.00),
                 verbose=1,
                 batch_iterator_train=BatchIterator(800),
                 max_epochs=270)
#
np.random.seed(17)
net0_clone = clone(net0)
net0_clone.fit(lay1preds_shared.get_value(), y)
#net0_clone.fit(X_encoded_shared.get_value(), y)

cv_by_hand = [(np.where(cvFolds != fold)[0], np.where(cvFolds == fold)[0])
               for fold in np.unique(cvFolds)]


foldPred = np.zeros((lay1preds_shared.get_value().shape[0], num_classes))
bags = 5
for iter in xrange(0,bags):
        for fold in xrange(0,np.unique(cvFolds).size):
            print("iter ", iter," fold ", fold )
            np.random.seed(iter + 56)
            net0_clone = clone(net0)
            net0_clone.fit(lay1preds_shared.get_value()[cv_by_hand[fold][0],:], y[cv_by_hand[fold][0]])
            foldPred[cv_by_hand[fold][1],:] += net0_clone.predict_proba(lay1preds_shared.get_value()[cv_by_hand[fold][1],:])
foldPredAdj = foldPred/bags            

# Load sample submission
#samp = pd.read_csv("/home/branden/Documents/kaggle/airbnb/sample_submission.csv") 
classMap = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/classMap.csv")
#Convert CV preds array to DataFrame with column names
foldPredFrame = pd.DataFrame(foldPred, columns=classMap.loc[:,"country_destination"])
# Add id column
foldPredFrame.insert(0, 'id', train_id)
# Save 
foldPredFrame.to_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_lay2_nn1.csv", index=False)


xgb1full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb1.csv")
xgb2full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb2.csv")
xgb3full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb3.csv")
xgb4full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb4.csv")
xgb5full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb5.csv")
xgb6full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb6.csv")
xgb7full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb7.csv")
xgb8full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb8.csv")
xgb9full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb9.csv")
xgb10full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb10.csv")
xgb11full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb11.csv")
xgb12full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb12.csv")
xgb13full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb13.csv")
xgb14full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_xgb14.csv")
nn1full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_nn1.csv")
nn2full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_nn2.csv")
nn3full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_nn3.csv")
nn4full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_nn4.csv")
glmnet1full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_glmnet1.csv")
rf1full = pd.read_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_rf1.csv")

xgb1full.drop("id", axis=1, inplace=True)
xgb2full.drop("id", axis=1, inplace=True)
xgb3full.drop("id", axis=1, inplace=True)
xgb4full.drop("id", axis=1, inplace=True)
xgb5full.drop("id", axis=1, inplace=True)
xgb6full.drop("id", axis=1, inplace=True)
xgb7full.drop("id", axis=1, inplace=True)
xgb8full.drop("id", axis=1, inplace=True)
xgb9full.drop("id", axis=1, inplace=True)
xgb10full.drop("id", axis=1, inplace=True)
xgb11full.drop("id", axis=1, inplace=True)
xgb12full.drop("id", axis=1, inplace=True)
xgb13full.drop("id", axis=1, inplace=True)
xgb14full.drop("id", axis=1, inplace=True)
nn1full.drop("id", axis=1, inplace=True)
nn2full.drop("id", axis=1, inplace=True)
nn3full.drop("id", axis=1, inplace=True)
nn4full.drop("id", axis=1, inplace=True)
glmnet1full.drop("id", axis=1, inplace=True)
rf1full.drop("id", axis=1, inplace=True)


lay1fullpreds = np.concatenate([xgb1full, xgb2full, xgb6full, xgb7full, xgb8full, xgb9full, xgb10full, xgb11full, xgb12full, xgb13full, xgb14full, nn1full, nn2full, nn3full, nn4full, rf1full, glmnet1full], axis=1).astype(np.float32)




np.random.seed(5)        
net1 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.0,
                 dense0_num_units=128,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dropout0_p=0.05,
                 dense0_nonlinearity=rectify,
#                 noise0_sigma=2,
                 dense1_num_units=128,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.05,
#                 dense2_num_units=64,
#                 dense2_W=GlorotUniform(),
#                 dense2_b = Constant(1.0),
#                 dense2_nonlinearity=rectify,
#                 dropout2_p=0.05,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 objective=regularization_objective,
#                 objective_lambda1=0.00000001,
#                 objective_loss_function=categorical_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0001), borrow=True),
#                 update_momentum=theano.shared(float32(0.001), borrow=True),
                 update_beta1=0.9,
                 update_beta2=0.99,
                 update_epsilon=1e-06,
                 on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.3, stop=0.05),
#                     AdjustVariable('update_momentum', start=0.001, stop=0.00299),
#                     EarlyStopping(patience=100),
                     ],               
                 
                 train_split=TrainSplit(eval_size=0.0),
                 verbose=1,
                 batch_iterator_train=BatchIterator(800),
                 max_epochs=400)


test_pred = np.zeros((lay1fullpreds.shape[0], num_classes))
for iter in xrange(0,bags):
    print(iter)
    np.random.seed(iter + 59)
    net1_clone = clone(net1)
    net1_clone.fit(lay1preds_shared.get_value(), y)
    test_pred += net1_clone.predict_proba(lay1fullpreds)
test_pred_adj = test_pred/bags    
    
    
# Load test data
test = pd.read_csv("/home/branden/Documents/kaggle/airbnb/test_users.csv")
test = test.sort_values('id').set_index(test.index)
    
#samp = pd.read_csv("/home/branden/Documents/kaggle/airbnb/sample_submission.csv") 
classMap = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/classMap.csv")
#Convert CV preds array to DataFrame with column names
testPredProbFrame = pd.DataFrame(test_pred_adj, columns=classMap.loc[:,"country_destination"])
# Add id column
testPredProbFrame.insert(0, 'id', test['id'])
# Save 
testPredProbFrame.to_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_lay2_nn1.csv", index=False)


#Taking the 5 classes with highest probabilities
#Adapted from https://www.kaggle.com/svpons/airbnb-recruiting-new-user-bookings/script-0-8655
ids = []  #list of ids
countries = []  #list of countries
for i in range(len(test['id'])):
    idx = test.loc[i,'id']
    ids += [idx] * 5
    countries += encoder.inverse_transform(np.argsort(test_pred_adj[i])[::-1])[:5].tolist()

#Generate submission
testPredFrame = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'class'])
testPredFrame['class'] = testPredFrame['class'].astype(np.int64)
testPredFrame = testPredFrame.merge(countryClasses, how="left", on="class")
testPredFrame.drop('class', axis=1, inplace=True)
testPredFrame.columns = ['id', 'country']
testPredFrame.to_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPreds_lay2_nn1.csv",index=False)















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


### START WITH DATA TRANSFORMATION CREATED IN R
ts1Trans = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/ts1_merge_v11.csv") 
cvFolds = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/cvFolds.csv")

t1nn = ts1Trans.loc[ts1Trans['filter']==0,]
s1nn = ts1Trans.loc[ts1Trans['filter']==2,]

labels = ts1Trans.loc[ts1Trans['filter']==0,'class']
t1nn_id = ts1Trans.loc[ts1Trans['filter']==0,'id']
encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.int32)

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
#
t1nn_conc = np.concatenate([t1nn_pca, t1nn_feats], axis=1)
##t1nn_conc = np.concatenate([t1nn_pca, t1nn_pca1], axis=1)
s1nn_conc = np.concatenate([s1nn_pca, s1nn_feats], axis=1)
#
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


# TRY AUTOENCODING
#
#layers0 = [('input', InputLayer),
#           ('noise0', GaussianNoiseLayer),        
#           ('dense0', DenseLayer),
#           ('dense1', DenseLayer),
#           ('encode0', DenseLayer),
#           ('denseOut0', DenseLayer),
#           ('denseOut1', DenseLayer),
#           ('output', DenseLayer)]
#
#
#
#np.random.seed(5)        
#ae0 = NeuralNet(layers=layers0,
#                 input_shape=(None, num_features),
#                 noise0_sigma=0.005,
#                 dense0_num_units=50,
#                 dense0_W=GlorotNormal(),
#                 dense0_b = Constant(1.0),
#                 dense0_nonlinearity=rectify,
#                 dense1_num_units=100,
#                 dense1_W=GlorotNormal(),
#                 dense1_b = Constant(1.0),
#                 dense1_nonlinearity=rectify,
#                 encode0_num_units = 200,
#                 encode0_W=GlorotNormal(),
#                 encode0_b = Constant(1.0),
#                 encode0_nonlinearity=rectify,
#                 denseOut0_num_units=100,
#                 denseOut0_W=GlorotNormal(),
#                 denseOut0_b = Constant(1.0),
#                 denseOut0_nonlinearity=rectify,
#                 denseOut1_num_units=50,
#                 denseOut1_W=GlorotNormal(),
#                 denseOut1_b = Constant(1.0),
#                 denseOut1_nonlinearity=rectify,
#                 output_num_units=num_features,
##                 output_nonlinearity=softmax,
##                 objective_loss_function=squared_error,
#                 
#                 update=adam,
#                 update_learning_rate=theano.shared(float32(0.0003), borrow=True),
##                 update_momentum=theano.shared(float32(0.001), borrow=True),
#                 update_beta1=0.9,
#                 update_beta2=0.999,
#                 update_epsilon=1e-06,
#                 on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.0003, stop=0.000001),
##                     AdjustVariable('update_momentum', start=0.001, stop=0.00299),
#                     EarlyStopping(patience=300),
#                     ],         
#                 regression=True,
#                 train_split=TrainSplit(eval_size=0.05),
#                 verbose=1,
#                 batch_iterator_train=BatchIterator(500),
#                 max_epochs=1000)
#
#np.random.seed(5)
#ae0_clone = clone(ae0)
#ae0_clone.fit(t1nn_feats_shared.get_value(), t1nn_feats_shared.get_value())
#
#
#def get_layer_by_name(net, name):
#    for i, layer in enumerate(net.get_all_layers()):
#        if layer.name == name:
#            return layer, i
#    return None, None
#    
#encode_layer, encode_layer_index = get_layer_by_name(ae0_clone, 'encode0')
#
#def encode_input(encode_layer, X):
#    return get_output(encode_layer, inputs=X).eval()
#
#X_encoded = encode_input(encode_layer, t1nn_feats)
#X_encoded_shared = theano.shared(X_encoded, borrow=True)
#
#num_classes = len(encoder.classes_)
#num_features = X_encoded.shape[1]

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

np.random.seed(15)        
net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.6,
                 dense0_num_units=100,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dropout0_p=0.3,
                 dense0_nonlinearity=rectify,
#                 noise0_sigma=2,
                 dense1_num_units=100,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.3,
#                 dense2_num_units=32,
#                 dense2_W=GlorotUniform(),
#                 dense2_b = Constant(1.0),
#                 dense2_nonlinearity=rectify,
#                 dropout2_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
#                 objective=regularization_objective,
#                 objective_lambda1=0.0000000001,
                 objective_loss_function=categorical_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0003), borrow=True),
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
                 max_epochs=100)

#np.random.seed(17)
#net0_clone = clone(net0)
#net0_clone.fit(t1nn_conc_shared.get_value(), y)
#net0_clone.fit(X_encoded_shared.get_value(), y)

cv_by_hand = [(np.where(cvFolds != fold)[0], np.where(cvFolds == fold)[0])
               for fold in np.unique(cvFolds)]


foldPred = np.zeros((t1nn_conc_shared.get_value().shape[0], num_classes))
bags = 20
folds = np.unique(cvFolds).size
for iter in xrange(0,bags):
        for fold in xrange(0,folds):
            np.random.seed(iter + 56)
            net0_clone = clone(net0)
            net0_clone.fit(t1nn_conc_shared.get_value()[cv_by_hand[fold][0],:], y[cv_by_hand[fold][0]])
            foldPred[cv_by_hand[fold][1],:] += net0_clone.predict_proba(t1nn_conc_shared.get_value()[cv_by_hand[fold][1],:])
foldPred = foldPred/bags            

# Load sample submission
#samp = pd.read_csv("/home/branden/Documents/kaggle/airbnb/sample_submission.csv") 
classMap = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/classMap.csv")
#Convert CV preds array to DataFrame with column names
foldPredFrame = pd.DataFrame(foldPred, columns=classMap.loc[:,"country_destination"])
# Add id column
foldPredFrame.insert(0, 'id', t1nn_id)
# Save 
foldPredFrame.to_csv("/home/branden/Documents/kaggle/airbnb/stack_models/cvPreds_nn4.csv", index=False)


np.random.seed(7)        
net1 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 inputDropout0_p=0.6,
                 dense0_num_units=100,
                 dense0_W=GlorotNormal(),
                 dense0_b = Constant(1.0),
                 dropout0_p=0.3,
                 dense0_nonlinearity=rectify,
#                 noise0_sigma=2,
                 dense1_num_units=100,
                 dense1_W=GlorotNormal(),
                 dense1_b = Constant(1.0),
                 dense1_nonlinearity=rectify,
                 dropout1_p=0.3,
#                 dense2_num_units=32,
#                 dense2_W=GlorotUniform(),
#                 dense2_b = Constant(1.0),
#                 dense2_nonlinearity=rectify,
#                 dropout2_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
#                 objective=regularization_objective,
#                 objective_lambda1=0.0000000001,
                 objective_loss_function=categorical_crossentropy,
                 
                 update=adam,
                 update_learning_rate=theano.shared(float32(0.0003), borrow=True),
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
                 max_epochs=100)



test_pred= np.zeros((s1nn_conc.shape[0], num_classes))
for iter in xrange(0,bags):
    np.random.seed(iter + 59)
    net1_clone = clone(net1)
    net1_clone.fit(t1nn_conc_shared.get_value(), y)
    test_pred += net1_clone.predict_proba(s1nn_conc)
test_pred_adj = test_pred/bags    


  

#samp = pd.read_csv("/home/branden/Documents/kaggle/airbnb/sample_submission.csv") 
classMap = pd.read_csv("/home/branden/Documents/kaggle/airbnb/data_trans/classMap.csv")
#Convert CV preds array to DataFrame with column names
testPredFrame = pd.DataFrame(test_pred_adj, columns=classMap.loc[:,"country_destination"]).set_index(s1nn.index)
# Add id column
testPredFrame.insert(0, 'id', s1nn['id'])
# Save 
testPredFrame.to_csv("/home/branden/Documents/kaggle/airbnb/stack_models/testPredsProbs_nn4.csv", index=False)


#NEED TO FIGURE OUT HOW TO PRODUCE ACTUAL SUBMISSION FILE IN PYTHON
##Taking the 5 classes with highest probabilities
#id_test = s1nn['id']
#ids = []  #list of ids
#cts = []  #list of countries
#for i in range(len(id_test)):
#    idx = id_test[i]
#    ids += [idx] * 5
#    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
#
##Generate submission
#sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
#sub.to_csv('sub.csv',index=False)






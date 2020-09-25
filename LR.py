# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        self.mean=[]
        self.standard_deviation=[]
        self.min_value=[]
        self.max_value=[]
        self.df_min_value=None
        self.df_max_value=None
    def __call__(self,df, is_train=False):
        result = df.copy()
        if is_train:
            for feature_name in df.columns:
                min_value=df[feature_name].min()
                max_value=df[feature_name].max()
                self.min_value.append(min_value)
                self.max_value.append(max_value)
                result[feature_name]=(df[feature_name]-min_value)/(max_value-min_value)
        else:
            i=0
            for feature_name in df.columns:
                min_value=self.min_value[i]
                max_value=self.max_value[i]
                result[feature_name]=(df[feature_name]-min_value)/(max_value-min_value)
                i+=1
        return result

def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''
    data_file=pd.read_csv(csv_path)
    data_file=data_file.drop(columns=[' shares'])
    data_file=scaler(data_file,is_train)
    data_file.apply(lambda x:x*x,axis=1)
    feature_matrix=data_file.to_numpy()

    print(feature_matrix.shape)
    return feature_matrix
    raise NotImplementedError

def get_features_test(csv_path,is_train=False,scaler=None):
    data_file=pd.read_csv(csv_path)
    data_file=scaler(data_file,is_train)
    data_file.apply(lambda x:x*x,axis=1)
    feature_matrix=data_file.to_numpy()
    return feature_matrix

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data_file=pd.read_csv(csv_path,usecols=[' shares'])
    target=data_file.to_numpy()
    return target
    raise NotImplementedError

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    m,n=feature_matrix.shape
    I=np.identity(n)
    feature_matrix_transpose=np.transpose(feature_matrix)
    temp=np.dot(feature_matrix_transpose,feature_matrix)
    t=C*m
    temp=np.add(temp,t*I)
    temp_inv=np.linalg.inv(temp)
    temp2=np.dot(feature_matrix_transpose,targets)
    analytical_weights=np.dot(temp_inv,temp2)
    return analytical_weights

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    predictions=np.dot(feature_matrix,weights)
    return predictions
    raise NotImplementedError

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    data_length,dimension=feature_matrix.shape
    loss=0.0
    Y_pred=get_predictions(feature_matrix,weights)
    Y_true=targets
    temp=np.subtract(Y_pred,Y_true)
    loss=np.sum(np.square(temp))
    return loss/data_length
    raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    result=int(pow(np.linalg.norm(weights),2))
    return result
    raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    loss=0.0
    loss=mse_loss(feature_matrix,weights,targets)+C*l2_regularizer(weights)
    return loss
    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=1e-8):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    n,d=feature_matrix.shape
    h=get_predictions(feature_matrix,weights)
    grad=np.dot(np.transpose(feature_matrix),np.subtract(h,targets))
    grad*=(2.0/n)
    t=2.0*C
    grad+=weights*t
    return grad
    raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    row,col=feature_matrix.shape
    indexes=np.random.choice(row,batch_size,replace=False)
    data=np.hstack((feature_matrix,targets))
    sampled_feature_matrix=data[indexes,0:col]
    sampled_target_matrix=data[indexes,col]
    sampled_target_matrix=sampled_target_matrix.reshape(-1,1)
    return sampled_feature_matrix,sampled_target_matrix
        

def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    weights=[0]*n
    weights=np.array(weights).reshape(-1,1)
    return weights
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    
    updated_weights=np.subtract(weights,gradients*lr)
    return updated_weights
    raise NotImplementedError

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=0.1,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    data_length,dimension=train_feature_matrix.shape
    weights = initialize_weights(dimension)
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''
    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

def save_csv_file(csv_path,test_targets):
    header=['shares']
    df=pd.DataFrame(data=test_targets,columns=header)
    df.index.name='instance_id'
    df.to_csv(csv_path)
if __name__ == "__main__":
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('train.csv',True,scaler), get_targets('train.csv')
    dev_features, dev_targets = get_features('dev.csv',False,scaler), get_targets('dev.csv')
    a_solution = analytical_solution(train_features, train_targets, C=1e-9)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    test_feature= get_features_test('test.csv',False,scaler)
    test_predictions=get_predictions(test_feature,a_solution)
    save_csv_file('my_predictions.csv',test_predictions)
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                            train_targets, 
                            dev_features,
                            dev_targets,
                            lr=0.12,
                            C=1e-11,
                            batch_size=32,
                            max_steps=2000000,
                            eval_steps=10000)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))    
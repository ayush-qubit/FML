import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import numpy as np

class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        self.mean=[]
        self.standard_deviation=[]
        self.min_value=[]
        self.max_value=[]
    def __call__(self,df, is_train=False):
        result = df.copy()
        if is_train:
            for feature_name in df.columns:
                min_value=df[feature_name].min()
                max_value=df[feature_name].max()
                self.min_value.append(min_value)
                self.max_value.append(max_value)
                #mean_value=df[feature_name].mean()
                #std_value=df[feature_name].std()
                #self.mean.append(mean_value)
                #self.standard_deviation.append(std_value)
                result[feature_name]=(df[feature_name]-min_value)/(max_value-min_value)
                #result[feature_name]=(df[feature_name]-mean_value)/(std_value)
        else:
            i=0
            for feature_name in df.columns:
                min_value=self.min_value[i]
                max_value=self.max_value[i]
                #mean_value=self.mean[i]
                #std_value=self.standard_deviation[i]
                result[feature_name]=(df[feature_name]-min_value)/(max_value-min_value)
                #result[feature_name]=(df[feature_name]-mean_value)/(std_value)
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
    #column_names=[' n_tokens_content',' num_hrefs',' num_self_hrefs',' num_imgs',' num_videos',' data_channel_is_lifestyle',' data_channel_is_entertainment',' data_channel_is_bus',' data_channel_is_socmed',' data_channel_is_tech',' self_reference_avg_sharess',' is_weekend',' global_rate_positive_words',' global_rate_negative_words',' avg_positive_polarity',' avg_negative_polarity',' abs_title_subjectivity']
    #column_names=get_best_feature()
    data_file=pd.read_csv(csv_path)
    data_file=data_file.drop(columns=[' shares'])
    data_file=scaler(data_file,is_train)
    data_file.apply(lambda x:x*x,axis=1)
    feature_matrix=data_file.to_numpy()
    #feature_matrix=scaler(feature_matrix)

    print(feature_matrix.shape)
    return feature_matrix
    raise NotImplementedError

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data_file=pd.read_csv(csv_path,usecols=[' shares'])
    '''for feature in data_file:
        data_file[feature]=(data_file[feature]-data_file[feature].min())/(data_file[feature].max()-data_file[feature].min())'''
    target=data_file.to_numpy()
    return target
    raise NotImplementedError

def get_features_test(csv_path,is_train=False,scaler=None):
    data_file=pd.read_csv(csv_path)
    #data_file=data_file.drop(columns=[' shares'])
    data_file=scaler(data_file,is_train)
    data_file.apply(lambda x:x*x,axis=1)
    feature_matrix=data_file.to_numpy()
    #feature_matrix=scaler(feature_matrix)
    #print(feature_matrix.shape)
    return feature_matrix

scaler=Scaler()
X_train=get_features('train.csv',True,scaler)
Y_train=get_targets('train.csv')
X_test=get_features_test('test.csv',is_train=False,scaler=scaler)
lm=LinearRegression()
lm.fit(X_train,Y_train)
Y_pred=lm.predict(X_test)
print(Y_pred)
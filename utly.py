import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
import tunable_param as tp
import global_fd as gfd

def getTrainLabels(): 
    train_labels = os.listdir(tp.train_path)
    train_labels.sort()
    return(train_labels)

def getDir(training_name):
    dir = os.path.join(tp.train_path, training_name)
    return dir

def read_im(file): 
    image = cv2.imread(file)
    image = cv2.resize(image, tp.fixed_size)
    return image

def fextract(image): 
    fv_hu_moments = gfd.fd_hu_moments(image)
    fv_histogram  = gfd.fd_histogram(image)
    fv_lbp        = gfd.fd_lbp(image)
    global_feature = np.hstack([fv_histogram, fv_lbp, fv_hu_moments])
    return global_feature

def encodeLabel(labels):
    le          = LabelEncoder()
    target      = le.fit_transform(labels)
    return target

def transf(global_features): 
    scaler = MinMaxScaler()
    rescaled_features = scaler.fit_transform(global_features)
    return rescaled_features
    
def check():
    print('\nChecking for dataset...')
    if os.path.isfile("output/data.h5") and os.path.isfile("output/labels.h5"):
        return False
    else:
        return True

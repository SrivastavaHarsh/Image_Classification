import h5py
import numpy as np
import tunable_param as tp

def saveH5(rescaled_features, target):
    h5f_data = h5py.File(tp.h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File(tp.h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

def readH5_data():
    h5f_data = h5py.File(tp.h5_data, 'r')
    global_features_string = h5f_data['dataset_1']
    global_features = np.array(global_features_string)
    h5f_data.close()
    return global_features

def readH5_label():
    h5f_label = h5py.File(tp.h5_labels, 'r')
    global_labels_string   = h5f_label['dataset_1']
    global_labels   = np.array(global_labels_string)
    h5f_label.close()
    return global_labels


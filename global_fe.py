import tunable_param as tp
import utly
import h5_handle as h5h
import global_var as gvr

def createDataset():
    print('\nCreating dataset...')
    # empty lists to hold feature vectors and labels
    global_features = []
    labels          = []

    print("[STATUS] Extracting Global Features...")
    # loop over the training data sub-folders
    for training_name in gvr.train_labels:
        dir = utly.getDir(training_name)
        current_label = training_name
        for x in range(1, tp.images_per_class + 1):
            file = dir + "/" + str(x) + ".jpg"
            image = utly.read_im(file)
            global_feature = utly.fextract(image)
            labels.append(current_label)
            global_features.append(global_feature)
        print("[STATUS] processed folder: {}".format(current_label))
    print("[STATUS] Completed Global Feature Extraction.")
    #print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
    #print("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    target = utly.encodeLabel(labels)
    # scale features in the range (0-1)
    rescaled_features = utly.transf(global_features)

    #print("[STATUS] target labels: {}".format(target))
    #print("[STATUS] target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5h.saveH5(rescaled_features, target)

    print('\nDataset created successfully.')

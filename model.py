import glob
import warnings
from sklearn.ensemble import RandomForestClassifier
import utly
import h5_handle as h5h
import tunable_param as tp
import display

def classify():
    warnings.filterwarnings('ignore')

    # get the training labels
    train_labels = utly.getTrainLabels()

    # import the feature vector and trained labels
    print('\nReading from dataset')
    global_features = h5h.readH5_data()
    global_labels   = h5h.readH5_label()

    print('\nTraining model...')
    # create the model - Random Forests
    clf  = RandomForestClassifier(n_estimators=tp.num_trees, random_state=tp.seed)
    # fit the training data to the model
    clf.fit(global_features, global_labels)
    print('\nModel trained.')

    print('\nRunning test...')
    # loop through the test images
    for file in glob.glob(tp.test_path + "/*.jpg"):
        image = utly.read_im(file)
        global_feature = utly.fextract(image)
        rgf = global_feature.reshape(-1,1)
        rescaled_feature = utly.transf(rgf)
        prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
        display.show(image, prediction)
    print('\nTest completed.')
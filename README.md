# Image_Classification
This repository (written in Python) contains code to classify images using global features such as shape, texture and color.</br>

## Build status : 
Basic classification model : completed.</br>
GUI : Under development.</br>
Enhanced classificastion model : under development.</br>

## Requirements :
### Libraries :
 cv2</br>
 matplotlib</br>
 skimage</br>
 scipy</br>
 h5py</br>
 numpy</br>
 glob</br>
 os</br>
 warnings</br>
 sklearn</br>

### *dataset* folder : 
There must be a folder named *dataset* which contains two subfolders : *test* and *train*.</br>
The *train* folder contains images that will be used to train our ML model inside their respective subfolders. Name of these subfolders will be used as label.</br>
The *test* subfolder contains images that will be classified using our model, once it is successfully trained.</br>

### *output* folder : 
There must also be an *output* folder which will store .h5 files.</br>

## Usage :
Run the *main.py* file to execute the program.</br>
  
## Keypoints about this project :
### Descriptors used :
  **For shape** :  Hu Moments</br>
  **For texture** : Local Binary Groups</br>
  **For colour** : Color Histogram</br>
  </br>
  The features extacted from the individual descriptors are simply added together to get the feature vector for an image.</br>
 
###  ML-Model used : 
**Random Decision Forest** : Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.</br>
   
### Dataset used : 
"will be uploaded soon"</br>


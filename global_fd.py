import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq

bins = 8

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    feature = cv2.HuMoments(cv2.moments(th)).flatten()
    return feature

def fd_histogram(image, mask=None):
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def fd_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    no_points = 8 * radius
    lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
    x = itemfreq(lbp.ravel())
    hist = x[:, 1]/sum(x[:, 1])
    return hist

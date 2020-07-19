import cv2
import matplotlib.pyplot as plt
import global_var as gvr

def show(image, prediction):
    cv2.putText(image, gvr.train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
### Please do not edit the following code block ###
import pkg_resources
import subprocess
import sys

required = {'opencv-python', 'scikit-learn', 'tensorflow', 'keras'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if len(missing) > 0:
    print('There are missing packages: {}.'.format(missing))
    for missing_package in missing:
        print(
            'Please wait for package installation and execute the program again.'
        )
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", missing_package
            if missing_package != 'tensorflow' else 'tensorflow==2.11.0'
        ])
    exit()

import cv2 
import os
from library import teachable_machine as tm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
### End of the code block ####

# A function reads an image from the dataset/test/ folder.
image = cv2.imread('./dataset/test/1.jpg')

# Adding salt and pepper noises to image.
# By adjusting the value of the "amount" parameter, you can observe the effect on the image,
# where higher values indicate a greater amount of noise added to the image.
#noise = tm.adjust_noise(image, amount=0)
# Plotting the image with added noise on the output tab
# tm.plot(noise)
# If you cannot plot, please enable the following code, it will save the image as 'saved_image.png' under Files tab by default.
# tm.plot(noise, save_image=True)


# # Adjusting the brightness of image.
# # By adjusting the value of the "beta" parameter, you can observe the effect on the image,
# # where higher values indicate a brighter image.
# bright = tm.adjust_brightness(image, beta=0)
# # Plotting the image with different brightness on the output tab
# tm.plot(bright)

# # Adjusting the covered areas of image.
# # By adjusting the value of the "percentage" parameter, you can observe the effect on the image,
# # where higher values indicate larger covered areas.
# partial = tm.adjust_partial(image, percentage=50)
# # Plotting the image with different covered areas on the output tab
# tm.plot(partial)

# # The code block does not necessitate comprehension.
# # Load a Teachable Machine model on replit
# model, class_names = tm.model_info()
# # Evaluate the model performance based on images from the folder of dataset/test/,
# # where 1.jpg-10.jpg are emergency vehicles and 11.jpg-20.jpg are normal vehicles
# test_image_dir = './dataset/test/'
# class_size = {'0 Emergency vehicle': 10, '1 Normal vehicle':10}
# truth, predicted = tm.prediction_result(test_image_dir, model, class_names, class_size)
# print("Ground truth is:", truth)
# print("Predicted result is", predicted)
# # To show confusion matrix
# tm.plot_confusion_matrix(truth, predicted)
# # If plot function is not working, please enable the following line, it will save the confusion matrix as 'confusion_matrix.png' under Files tab
# # tm.plot_confusion_matrix(truth, predicted, save_image=True)

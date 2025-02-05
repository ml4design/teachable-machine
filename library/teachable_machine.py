from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Adding salt and pepper noise
# Salt and pepper noise is a type of image noise that affects digital images.
# It is called salt and pepper noise because the affected pixels appear as isolated white and black dots,
# scattered randomly across the image, like grains of salt and pepper.
def adjust_noise(image, amount):
    s_vs_p = 0.5
    amount /= 1000
    out = np.copy(image)

    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = tuple(
        [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]])
    # coords = [tuple(each) for each in np.array(coords).reshape(-1, 2)]
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = tuple([
        np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]
    ])
    out[coords] = 0

    return out


# Brightness adjustments
# Increasing (/ decreasing) the β value will add (/ subtract) a constant value to every pixel.
def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta * 2.5)


# Showing partial images
# The adjustment turns the left part of images to black
def adjust_partial(image, percentage):
    partial = image.copy()
    partial[:, int(partial.shape[0] * (100 - percentage) / 100):, :] = 0
    return partial


# Plot or save image
def plot(image, save_image=False, image_name='saved_image.png'):
    if save_image:
        cv2.imwrite(image_name, image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()


# Load Teachable Machine model and class names
def model_info(model_path='./model/keras_model.h5',
               class_path='./model/labels.txt'):
    # Load the model
    model = load_model(model_path, compile=False)
    # Load the labels
    class_names = open(class_path, "r").readlines()

    return model, class_names


# predict the input image class through Teachable machine model
def predict(image, model, class_names, print_info=False):
    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    if print_info:
        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:",
              str(np.round(confidence_score * 100))[:-2], "%")

    return class_name[2:]


# to predict generate predicted result
def prediction_result(test_image_dir, model, class_names, class_size):
    test_image_dir = './dataset/test/'
    truth = []
    for class_name in class_names:
        truth += [
            class_name[2:].replace('\n', '')
            for _ in range(class_size[class_name.replace('\n', '')])
        ]
    predicted = []
    for item in sorted(os.listdir(test_image_dir),
                       key=lambda x: int(x.replace('.jpg', ''))):
        image = cv2.imread(test_image_dir + item)
        predicted.append(predict(image, model, class_names).replace('\n', ''))

    return truth, predicted

# To plot confusion matrix
def plot_confusion_matrix(truth, predicted, save_image=False, image_name='confusion_matrix.png'):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay.from_predictions(truth,
                                            predicted,
                                            cmap='Blues',
                                            ax=ax)
    if save_image:
        plt.savefig(image_name)
    else:
        plt.show()
    
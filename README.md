# Facial Emotion Recognition

This project aims to recognize facial emotions using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The model is trained on a dataset containing images of various facial expressions and can predict emotions in real-time using a webcam.

## Overview

Facial Emotion Recognition involves identifying human emotions from facial expressions. This project utilizes a CNN model implemented with TensorFlow and Keras to classify emotions from images and real-time webcam feed.

## Dataset

The training dataset consists of images categorized into seven emotions:

* Happy: 7215 images
* Sad: 4830 images
* Fear: 4097 images
* Surprise: 3171 images
* Neutral: 4965 images
* Angry: 3995 images
* Disgust: 436 images

The testing dataset consists of images categorized into the same seven emotions:

* Happy: 1774 images
* Sad: 1247 images
* Fear: 1024 images
* Surprise: 831 images
* Neutral: 1233 images
* Angry: 958 images
* Disgust: 111 images

## Model

The Convolutional Neural Network (CNN) used in this project for facial emotion recognition is designed to classify facial emotions from images. The network includes several key components and strategies to enhance its performance and prevent overfitting.

A custom learning rate schedule adjusts the learning rate during training to ensure optimal learning. The learning rate starts at 0.0005, is reduced to 0.00025 after 20 epochs, and further decreased to 0.00005 after 30 epochs. This gradual reduction helps in fine-tuning the model, allowing it to converge more effectively during later stages of training.

The model begins with an initialization as a sequential model. The first layer consists of a convolutional layer with 64 filters of size 3x3, using padding to preserve the input dimensions. This is followed by batch normalization to normalize the activations, ReLU activation for non-linearity, max pooling to reduce spatial dimensions, and dropout to randomly set 30% of the inputs to zero, thus preventing overfitting.

The second convolutional layer increases the number of filters to 128, maintaining the same structure as the first layer with batch normalization, ReLU activation, max pooling, and dropout. This layer further abstracts the features learned from the input images.

The third convolutional layer continues with 256 filters and follows a similar structure but with a higher dropout rate of 40%. This layer aims to capture more complex features while further reducing the chances of overfitting.

The fourth convolutional layer, with 512 filters, also follows the established pattern of batch normalization, ReLU activation, max pooling, and a 40% dropout rate. This layer deepens the network, allowing it to learn more intricate patterns and details from the images.

Following the convolutional layers, a global average pooling layer is used to reduce each feature map to a single value by averaging the values, significantly reducing the number of parameters and thereby simplifying the model.

A fully connected layer with 512 units is then added, incorporating L2 regularization to further prevent overfitting. This layer also includes batch normalization, ReLU activation, and a 50% dropout rate, ensuring robust learning while mitigating overfitting risks.

The final output layer consists of 7 units, corresponding to the seven emotion classes. It uses a softmax activation function to convert the outputs into probabilities, facilitating the classification of the input images into one of the seven emotion categories.

The model is compiled using the Adam optimizer with an initial learning rate of 0.0005. The loss function used is categorical cross-entropy, suitable for multi-class classification tasks, and the accuracy metric is used to evaluate the model's performance.

Throughout training, a learning rate scheduler callback is used to adjust the learning rate according to the predefined schedule, ensuring that the model learns efficiently and effectively. This well-structured CNN architecture, combining convolutional layers, batch normalization, dropout, and fully connected layers, is designed to efficiently learn and classify facial emotions from images.

The CNN model is built using TensorFlow and Keras. The model architecture is saved in JSON format for easy loading and deployment.

## Usage

### Training the Model

To train the model with the dataset, run: 

```
jupyter notebook training_Model.ipynb
```

### Real-time Emotion Recognition

To perform real-time emotion recognition using a webcam, run:

```
jupyter notebook testing_Model.ipynb
```

The script uses `cv2.CascadeClassifier('haarcascade_frontalface_default.xml')` to detect faces and then predicts emotions on the detected faces.

## Acknowledgements

* This project uses the TensorFlow and Keras libraries for building and training the CNN model.
* The Haar Cascade for face detection is provided by OpenCV.

_________________________________________________________________________________________________________________________

---
title: "Machine Learning Flash"
teaching: 15
exercises: 15
questions:
- "How can I use scikit learn to apply machine learning?"
objectives:
- "Run a SVM classifier on the MNIST digits"
- "Make a prediction for an arbitrary set of images of a digit."
keypoints:
- "flatten input dataset as the SVM is unaware of the idea of an image"
- "Split your data 50/50 and train on the first half."
- "Predict the other half."
- "Produce a confusion matrix to check the quality of the learning."
- "Plot some images and their predicted values."
---
## `scikit-learn` is one of the most widely used scientific machine learning library in Python.

*   Commonly called `sklearn`.
*   The Jupyter Notebook will render plots inline if we ask it to using a "magic" command.

~~~
%matplotlib inline
import matplotlib.pyplot as plt
~~~
{: .python}

*   import scikit-learn

~~~
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
~~~
{: .python}

* load the data

~~~
digits = datasets.load_digits()
~~~
{: .python}


## Get the lay of the land

*   combine two lists using the zip function for easier handling inside the plotting loop
	* note: `target` refers to a numerical representation of the labels

~~~
images_and_labels = list(zip(digits.images, digits.target))
~~~
{: .python}


*   create several subplots to draw the first four items in the dataset as well as their actual label

~~~
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
~~~
{: .python}



## Flatten the input images 

*   The inputs are 8x8 grayscale images
*   produce a flat array of 64 pixel values so that each pixel corresponds to a column/observable for the classifier later on

~~~
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
~~~
{: .python}

## Train the Classifier

*   in this example, we use a support vector machine provided within `sklearn`

~~~
classifier = svm.SVC(gamma=0.001)
~~~
{: .python}

*   we train the classifier on half the dataset

~~~
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
~~~
{: .python}


## Machine Learning in action

* Let's do the prediction on the remaining half of the dataset

~~~
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
~~~
{: .python}

* compute some metrics/statistics on the quality of the prediction, misprediction rates etc.

~~~
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
~~~
{: .python}


## Let the machine speak

* if satisfied with the above, we can predict some images

~~~
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
~~~

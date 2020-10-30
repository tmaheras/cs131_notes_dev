---
title: Visual recognition
keywords: (insert comma-separated keywords here)
order: 12 # Lecture number for 2020
---

Table of Contents
12.1 Introduction to Object Recognition
Visual Recognition Tasks and Applications
Challenges 
12.2 k-Nearest Neighbors
Machine Learning Framework
Nearest Neighbors Classifier
K-Nearest Neighbors
Challenges
Bias and Variance
Bias Variance Tradeoff
No Free Lunch Theorem
12.3 A Simple Object Recognition Pipeline
A Simple Image Classifier
Pipeline
Feature Extraction
Test Image Classification
Performance Evaluation









12.1: Introduction to Object Recognition
Visual Recognition Tasks and Applications
In general, the goal is to design an artificial visual system that can mimic human vision capabilities.
Tasks
Classification. Classify an entire image as belonging to a certain category or not (e.g. does it contain a particular object or not).
Detection. Identifying if some object is in the image, and where it is in the image. Essentially, this task is a combination of classification and localization and can be more sophisticated by detecting object semantic and geometric attributes such as relative distances and sizes.
Applications:
Security
Computational Photography
Assistive Driving 
Single Instance Recognition. Recognizing whether a particular instance of an object exists in the image and where in the image is it located.
Applications:
Landmark Detection and GPS
Activity or Event Recognition. Being able to recognize what is happening in an image and what activities are the subjects of the image engaged in.
Applications:
Human-Robot Interaction
Sign Language
Ambient Intelligence
Sport Analysis

Challenges
There are many challenges one will encounter when building algorithms that can classify human activities and events, detect and localize objects, estimate semantic attributes and classify objects. These challenges include:
A large number of categories for object classification. For example, ImageNet, a large dataset for object recognition, contains over 14M images and over 20k categories

Viewpoint Variation: The same image will appear to look quite different depending on the angle at which it was taken. As shown below, the geometric positions of the statue’s facial features vary greatly depending on the position of the camera.

Illumination: The position, intensity and color of the light will impact how an object looks

Scale: It is challenging to estimate and adjust for scale due to the location of the camera, differences in size of other objects in the image and instances of the same object category may vary greatly in size


Deformation: An object may have certain characteristics that make its body not rigid which may result in the object’s shape or patterns differing between images. For example, a cat does not have a rigid body and may take many different shapes


Occlusion: Objects are covered by other objects in an image and are not fully visible


Background Clutter: For objects that blend in with the background, it is difficult to segment and accurately classify the object
 

Intra-Class Variation: Instances within the same class can look quite different from each other and the computer vision algorithms must be able to account for this variation


12.2: k-Nearest Neighbors

Machine Learning Framework

Training: given a training set of labeled examples  estimate the prediction function  by minimizing the prediction error on the training set.

Testing: apply  to a never before seen test example  and output predicted value .

Classification

Assign input vector to one of two or more classes

Any decision rule divides input space into decision regions separated by decision boundaries. 

Any input image can be mapped as a point in this space: we extract two features in this case  and 


Nearest Neighbor Classifier

Assign a label of nearest training data point to each test data point.

We compare the test instance with every set in the training set. We chose an example that is closest in distance to the test image. 

Above process can be seen as partitioning the feature space. 

For example, if the input image is represented by two features  and . This process is similar to dividing this space into regions. So if the red dots and black dots below are training samples, the boundaries represent the area in the plane that is closest to each training sample. When we see a new test sample we can map it to one of the boundaries: 

This same idea works regardless of the dimensionality. The example below is a 3D analogue of the 2D version above. 

K-nearest neighbors
Algorithm (training):
Store all training data points  with their corresponding category labels 
Algorithm (testing):
We are given a new test point .
Compute distance to all training data points.
Select  training points closest to .
Assign  to label  that is most common among the  nearest neighbors.
Distance measurement (one example):
Euclidean distance:
 

Consider the following as training samples. 

So if , and black point is our test sample, we select the closest neighbor and assign it a classification of red cross:

Similarly, if , we extract the 3 nearest neighbors. Since now we have green circles and red crosses we select the classification with the most common label. In this case there’s 3 green circles and 1 red cross so we assign a classification of green circle this time. 
  
The decision boundaries produced by k-nearest neighbors are directly established from the data. The algorithm is very flexible in capturing complex relations between the training data points, and yet it is one of the simplest classification algorithms. Because it is relatively simple, it is a good option to try first when tackling a classification problem. However, it is important to understand the drawbacks and challenges. 

k-Nearest Neighbors Challenges
Choosing k
If k is too small, the decisions may be too sensitive to noisy data points. For example, a training sample may have the wrong label. 
If k is too large, the neighborhood may include points from other classes. 

Ex: Boundaries that can be obtained by changing the value of k from 1 to 15 on a dataset with 3 classes.

Note that when k=1, the decision boundaries are highly complex. Whereas when k=15, the boundaries are simpler. This is usually good because it means they aren’t overly sensitive to noise in the data.


Decreasing k yields more complex decision boundaries, producing smaller regions. More complex models yield smaller train error, but test error tends to be higher. Increasing k yields simpler, smoother decision boundaries (less complex model). We want to choose a k that minimizes testing error.

So how do we choose k? The solution is cross validation.
For each value of k in the nearest neighbor algorithm:
Create multiple train/test splits of the data. By creating different sets of training and testing data, we create multiple scenarios under which to validate our algorithm.
Measure performance for each split.
Average performance over all splits.
Select k with the best average performance.




Euclidean Measurement
When dimensionality of vectors is high, sometimes Euclidean distance measurement can produce counterintuitive results.

Ex:



Solve this by normalizing vectors to be of unit length

Curse of Dimensionality


Generalization:

Given a training set of images for which I have labels that are known...
And a new test set for which to predict the labels...
How well does the model I train on the training data generalizes from that data into the test set?

What’s the error of my classifier in the testing set after I have trained in the training set?

Takeaway: Can one know beforehand which of the potential classifiers one can use, and will that classifier produce the lowest generalization error or lowest test error in the testing set.


Bias and Variance:

	Two Components of Generalization Error:

Bias: how much the average model over all training sets differ from the true model?

Variance: how much models estimated from different training sets differ from each other

	These result in:

Underfitting: model is too “simple” to represent all the relevant class characteristics
High bias and low variance
High training error and high test error

Overfitting: model is too “complex” and fits irrelevant characteristics (noise) in the data
Low bias and high variance
Low training error and high test error

	
To put these ideas into visual context, we will create a simplified case:

		Suppose we are fitting an equation to the red points in this plot:
The True equation is green line
(To be estimated by the red points)
What models can best fit the true equation?


	
	Linear model?
Too few parameters 
two coefficients in the equation
Too simple to fit the red points
Underfitting

A model with too few parameters is INACCURATE because of a LARGE bias.
It doesn’t have enough flexibility or capacity to fit the data.




High degree polynomial model:
Too many parameters.
Our model changes dramatically with each red point
Overfitting

A model with too MANY parameters is INACCURATE because of a large variance.
Too sensitive to the specific training set we are using, and therefore, a very large variance.


Bias versus Variance Tradeoff

	Best explained by this plot:
	


Key observations:
The more complex the model, the lower the Bias.
The more complex the model, the HIGHER the variance
There is a sweet spot between these two.

We can do model selection to select the model complexity that will obtain the lowest total error. HOWEVER, how do we know what other classifiers have the lowest possible error?


No Free Lunch Theorem:

Unfortunately we can’t really tell beforehand which classifier will have the lowest generalization error. 

We can only get generalization through some assumptions.

	3 Kinds of Errors:
Inherent: unavoidable
Bias: due to over-simplifications
Variance: due to inability to perfectly estimate parameters from limited data

How to reduce these errors?:
Reduce Variance:
Choose a simpler classifier
Regularize the parameters
Get more training data
Reduce Bias:
Choose a more complex classifier
Increase the amount of features / parameters
Get more training data


Machine Learning Methods Final Tips:

Know your data
How much supervision do you have?
How many training examples can you afford?
How noisy?
	Know your goal:
Affects your choices of representation
Affects your choices of learning algorithms
Affects your choices of evaluation metrics
	
	Understand the math behind each machine learning algorithm you are choosing!



12.3: A Simple Object Recognition Pipeline
A Simple Image Classifier

For this example of an object recognition pipeline, we will use a simple image classifier.
-We therefore will be using a prediction function that takes feature representations of images as inputs and yields classifications of their contents as outputs.


Pipeline

The pipeline contains the following steps:
First, we ingest a collection of training images and extract image features from them
We train the classifier using the image features of the images as well as their (ground-truth) training labels.
After training this classifier, we can now use it to predict classifications for unseen testing images:
First, we extract features from the test images
Then we feed those features to our trained classifier to output the prediction for the classification of the object.

This pipeline is illustrated below:




Feature Extraction

Here are a few different methods for extracting these image features from training and testing images:

Extracting color features, for example by creating a color histogram to represent the frequency with which each color appears in the image.
These features are translation, scale, and rotation invariant
However, they are not occlusion invariant
Using features that capture the object’s shape
These features are also translation and rotation invariant, but not occlusion invariant
Extracting features that capture local shape
This can be done with keypoint detection, followed by extracting the shape of each keypoint
Such features are only translation and scale invariant
Using filter banks to extract texture-based features
The resulting features are translation invariant


Test Image Classification

When training is complete and features have been extracted, the trained classifier is then used to classify each new test image.
-For example, if nearest-neighbor is used, the test image will be classified with the same label as its closest neighbor in the training set. In the example below, the test image would be classified as a blue square:




Performance Evaluation

After labeling the testing examples using the classifier, we can evaluate the algorithm’s performance.
-We do so by calculating the percentage of correctly classified images from the testing set. 
-Classification accuracy for certain images can vary significantly based on the feature extraction method used.


edit

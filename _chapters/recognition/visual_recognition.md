---
title: Visual recognition
keywords: (insert comma-separated keywords here)
order: 12 # Lecture number for 2020
---

## Table of Contents

## **12.1 Introduction to Object Recognition**

- Visual Recognition Tasks and Applications
- Challenges

## **12.2 k-Nearest Neighbors**

- Machine Learning Framework
- Nearest Neighbors Classifier
- K-Nearest Neighbors
  - Challenges

- Bias and Variance
  - Bias Variance Tradeoff
- No Free Lunch Theorem

## **12.3 A Simple Object Recognition Pipeline**

- A Simple Image Classifier
- Pipeline
- Feature Extraction
- Test Image Classification
- Performance Evaluation

#


# **12.1: Introduction to Object Recognition**

**Visual Recognition Tasks and Applications**

In general, the goal is to design an artificial visual system that can mimic human vision capabilities.

Tasks

1. **Classification.** Classify an entire image as belonging to a certain category or not (e.g. does it contain a particular object or not).
2. **Detection.** Identifying if some object is in the image, and where it is in the image. Essentially, this task is a combination of classification and localization and can be more sophisticated by detecting object semantic and geometric attributes such as relative distances and sizes.

  - Applications:
    - Security
    - Computational Photography
    - Assistive Driving

1. **Single Instance Recognition.** Recognizing whether a particular instance of an object exists in the image and where in the image is it located.

- Applications:
  - Landmark Detection and GPS

1. **Activity or Event Recognition.** Being able to recognize what is happening in an image and what activities are the subjects of the image engaged in.

- Applications:
  - Human-Robot Interaction
  - Sign Language
  - Ambient Intelligence
  - Sport Analysis

**Challenges**

There are many challenges one will encounter when building algorithms that can classify human activities and events, detect and localize objects, estimate semantic attributes and classify objects. These challenges include:

1. **A large number of categories for object classification.** For example, ImageNet, a large dataset for object recognition, contains over 14M images and over 20k categories

![](RackMultipart20201030-4-6br838_html_1a78191c728f59f2.png)

1. **Viewpoint Variation:** The same image will appear to look quite different depending on the angle at which it was taken. As shown below, the geometric positions of the statue&#39;s facial features vary greatly depending on the position of the camera.

![](RackMultipart20201030-4-6br838_html_43652a4ac1f63d93.png)

1. **Illumination:** The position, intensity and color of the light will impact how an object looks

![](RackMultipart20201030-4-6br838_html_bd814cfc3c514019.png)

1. **Scale:** It is challenging to estimate and adjust for scale due to the location of the camera, differences in size of other objects in the image and instances of the same object category may vary greatly in size

![](RackMultipart20201030-4-6br838_html_cc00fe7e822a4918.png)

1. **Deformation:** An object may have certain characteristics that make its body not rigid which may result in the object&#39;s shape or patterns differing between images. For example, a cat does not have a rigid body and may take many different shapes

![](RackMultipart20201030-4-6br838_html_6b42d9ba0a0bdcf2.jpg) ![](RackMultipart20201030-4-6br838_html_c62142ccb706ca0e.png)

1. **Occlusion:** Objects are covered by other objects in an image and are not fully visible

![](RackMultipart20201030-4-6br838_html_b5c708e172c76cbf.png)

1. **Background Clutter:** For objects that blend in with the background, it is difficult to segment and accurately classify the object

![](RackMultipart20201030-4-6br838_html_8cc6d025ae5a438e.png)

1. **Intra-Class Variation:** Instances within the same class can look quite different from each other and the computer vision algorithms must be able to account for this variation

![](RackMultipart20201030-4-6br838_html_d763f55885cd20af.png)

#


# **12.2: k-Nearest Neighbors**

**Machine Learning Framework**

![](RackMultipart20201030-4-6br838_html_f6d9cb6bb9ada21e.png)

**Training** : given a _training set_ of labeled examples ![](RackMultipart20201030-4-6br838_html_fcd76b1bc5c3250b.gif) estimate the prediction function ![](RackMultipart20201030-4-6br838_html_f476b959a6eeb73f.png) by minimizing the prediction error on the training set.

**Testing** : apply ![](RackMultipart20201030-4-6br838_html_f476b959a6eeb73f.png) to a never before seen test example ![](RackMultipart20201030-4-6br838_html_93594d5e058ef2e4.png) and output predicted value ![](RackMultipart20201030-4-6br838_html_50636a9d849f37aa.png).

**Classification**

Assign input vector to one of two or more classes

Any decision rule divides input space into _decision regions_ separated by _decision boundaries_.

Any input image can be mapped as a point in this space: we extract two features in this case ![](RackMultipart20201030-4-6br838_html_cc374ff06aa07ca2.png) and ![](RackMultipart20201030-4-6br838_html_833327733580003a.png)

![](RackMultipart20201030-4-6br838_html_c555ebad878f45fb.png)

**Nearest Neighbor Classifier**

Assign a label of nearest training data point to each test data point.

![](RackMultipart20201030-4-6br838_html_51bff2b1c635554f.png)

We compare the test instance with every set in the training set. We chose an example that is closest in distance to the test image.

Above process can be seen as partitioning the feature space.

For example, if the input image is represented by two features ![](RackMultipart20201030-4-6br838_html_cc374ff06aa07ca2.png) and ![](RackMultipart20201030-4-6br838_html_833327733580003a.png). This process is similar to dividing this space into regions. So if the red dots and black dots below are training samples, the boundaries represent the area in the plane that is closest to each training sample. When we see a new test sample we can map it to one of the boundaries:

![](RackMultipart20201030-4-6br838_html_a4113ac01f9848fc.png)

This same idea works regardless of the dimensionality. The example below is a 3D analogue of the 2D version above.

![](RackMultipart20201030-4-6br838_html_4c6bc9be8930496a.png)

**K-nearest neighbors**

**Algorithm (training):**

1. Store all training data points ![](RackMultipart20201030-4-6br838_html_99e0f0a18e20cbb8.png) with their corresponding category labels ![](RackMultipart20201030-4-6br838_html_58887a01d301a586.png)

**Algorithm (testing):**

1. We are given a new test point ![](RackMultipart20201030-4-6br838_html_93594d5e058ef2e4.png).
2. Compute distance to all training data points.
3. Select ![](RackMultipart20201030-4-6br838_html_ef2d436003acbec4.png) training points closest to ![](RackMultipart20201030-4-6br838_html_93594d5e058ef2e4.png).
4. Assign ![](RackMultipart20201030-4-6br838_html_93594d5e058ef2e4.png) to label ![](RackMultipart20201030-4-6br838_html_fd354eb8f654d5c1.png) that is most common among the ![](RackMultipart20201030-4-6br838_html_ef2d436003acbec4.png) nearest neighbors.

**Distance measurement** (one example) **:**

- Euclidean distance:

![](RackMultipart20201030-4-6br838_html_1b462a84935b74a0.png)

Consider the following as training samples.

![](RackMultipart20201030-4-6br838_html_eed5d6a55ce6303b.png)

So if ![](RackMultipart20201030-4-6br838_html_832fba4ec542b1c6.png), and black point is our test sample, we select the closest neighbor and assign it a classification of red cross:

![](RackMultipart20201030-4-6br838_html_7d67f195590f8d.png)

Similarly, if ![](RackMultipart20201030-4-6br838_html_b8fcb6837eea59f9.png), we extract the 3 nearest neighbors. Since now we have green circles and red crosses we select the classification with the most common label. In this case there&#39;s 3 green circles and 1 red cross so we assign a classification of green circle this time.

![](RackMultipart20201030-4-6br838_html_903ae2b1cc8a8a31.png)

The decision boundaries produced by k-nearest neighbors are directly established from the data. The algorithm is very flexible in capturing complex relations between the training data points, and yet it is one of the simplest classification algorithms. Because it is relatively simple, it is a good option to try first when tackling a classification problem. However, it is important to understand the drawbacks and challenges.

**k-Nearest Neighbors Challenges**

**Choosing k**

- If k is too small, the decisions may be too sensitive to noisy data points. For example, a training sample may have the wrong label.
- If k is too large, the neighborhood may include points from other classes.

Ex: Boundaries that can be obtained by changing the value of k from 1 to 15 on a dataset with 3 classes.

![](RackMultipart20201030-4-6br838_html_d9bb1a4e631bb9af.png)

Note that when k=1, the decision boundaries are highly complex. Whereas when k=15, the boundaries are simpler. This is usually good because it means they aren&#39;t overly sensitive to noise in the data.

![](RackMultipart20201030-4-6br838_html_6e1d50c4f0a0aeed.png)

Decreasing k yields more complex decision boundaries, producing smaller regions. More complex models yield smaller train error, but test error tends to be higher. Increasing k yields simpler, smoother decision boundaries (less complex model). We want to choose a k that minimizes testing error.

So how do we choose k? The solution is **cross validation.**

- For each value of k in the nearest neighbor algorithm:
  - Create multiple train/test splits of the data. By creating different sets of training and testing data, we create multiple scenarios under which to validate our algorithm.
  - Measure performance for each split.
  - Average performance over all splits.
- Select k with the best average performance.

![](RackMultipart20201030-4-6br838_html_a1ec3509ee183b8.png)

**Euclidean Measurement**

When dimensionality of vectors is high, sometimes Euclidean distance measurement can produce counterintuitive results.

Ex:

![](RackMultipart20201030-4-6br838_html_6070fda90b60024d.png)

Solve this by normalizing vectors to be of unit length

**Curse of Dimensionality**

**Generalization:**

![](RackMultipart20201030-4-6br838_html_84759fc8b65c8a44.png)

1. Given a training set of images for which I have labels that are known...
2. And a new test set for which to predict the labels...
3. How well does the model I train on the training data generalizes from that data into the test set?

_What&#39;s the error of my classifier in the testing set after I have trained in the training set?_

**Takeaway:** Can one know beforehand which of the potential classifiers one can use, and will that classifier produce the lowest generalization error or lowest test error in the testing set.

**Bias and Variance:**

Two Components of Generalization Error:

- **Bias:** _how much the average model over all training sets differ from the true model?_

- **Variance** : _how much models estimated from different training sets differ from each other_

These result in:

- **Underfitting** : _model is too &quot;simple&quot; to represent all the relevant class characteristics_
  - High bias and low variance
  - High training error and high test error

- **Overfitting:** _model is too &quot;complex&quot; and fits irrelevant characteristics (noise) in the data_
  - Low bias and high variance
  - Low training error and high test error

To put these ideas into visual context, we will create a simplified case:

Suppose we are fitting an equation to the red points in this plot:

- The True equation is green line
  - (To be estimated by the red points)
- What models can best fit the true equation?

![](RackMultipart20201030-4-6br838_html_f7d6d14eded36982.png)

Linear model?

- Too few parameters
  - two coefficients in the equation
  - Too simple to fit the red points
- **Underfitting**

_A model with too few parameters is INACCURATE because of a LARGE bias._

_It doesn&#39;t have enough flexibility or capacity to fit the data._

![](RackMultipart20201030-4-6br838_html_d8e95c1759ea4c4c.png)

High degree polynomial model:

- Too many parameters.
  - Our model changes dramatically with each red point
- **Overfitting**

- _A model with too MANY parameters is INACCURATE because of a large variance._
  - _Too sensitive to the specific training set we are using, and therefore, a very large variance._

**Bias versus Variance Tradeoff**

Best explained by this plot:

![](RackMultipart20201030-4-6br838_html_b13ee7de3bcd315d.png)

**Key observations:**

- The more complex the model, the lower the Bias.
- The more complex the model, the HIGHER the variance
- There is a sweet spot between these two.

We can do model selection to select the model complexity that will obtain the lowest total error. HOWEVER, how do we know what other classifiers have the lowest possible error?

**No Free Lunch Theorem:**

Unfortunately we can&#39;t really tell beforehand which classifier will have the lowest generalization error.

We can only get generalization through some assumptions.

**3 Kinds of Errors:**

- _Inherent:_ unavoidable
- _Bias:_ due to over-simplifications
- _Variance:_ due to inability to perfectly estimate parameters from limited data

**How to reduce these errors?:**

- _Reduce Variance:_
  - Choose a simpler classifier
  - Regularize the parameters
  - Get more training data
- _Reduce Bias:_
  - Choose a more complex classifier
  - Increase the amount of features / parameters
  - Get more training data

**Machine Learning Methods Final Tips:**

Know your data

- How much supervision do you have?
- How many training examples can you afford?
- How noisy?

Know your goal:

- Affects your choices of representation
- Affects your choices of learning algorithms
- Affects your choices of evaluation metrics

_Understand the math behind each machine learning algorithm you are choosing!_

#


# **12.3: A Simple Object Recognition Pipeline**

**A Simple Image Classifier**

For this example of an object recognition pipeline, we will use a simple image classifier.

-We therefore will be using a prediction function that takes feature representations of images as inputs and yields classifications of their contents as outputs.

**Pipeline**

The pipeline contains the following steps:

1. First, we ingest a collection of training images and extract image features from them
2. We train the classifier using the image features of the images as well as their (ground-truth) training labels.
3. After training this classifier, we can now use it to predict classifications for unseen testing images:
  1. First, we extract features from the test images
  2. Then we feed those features to our trained classifier to output the prediction for the classification of the object.

This pipeline is illustrated below:

![](RackMultipart20201030-4-6br838_html_10df471cf85e2cbe.png)

**Feature Extraction**

Here are a few different methods for extracting these image features from training and testing images:

- Extracting color features, for example by creating a color histogram to represent the frequency with which each color appears in the image.
  - These features are translation, scale, and rotation invariant
  - However, they are not occlusion invariant
- Using features that capture the object&#39;s shape
  - These features are also translation and rotation invariant, but not occlusion invariant
- Extracting features that capture _local_ shape
  - This can be done with keypoint detection, followed by extracting the shape of each keypoint
  - Such features are only translation and scale invariant
- Using filter banks to extract texture-based features
  - The resulting features are translation invariant

**Test Image Classification**

When training is complete and features have been extracted, the trained classifier is then used to classify each new test image.

-For example, if nearest-neighbor is used, the test image will be classified with the same label as its closest neighbor in the training set. In the example below, the test image would be classified as a blue square:

![](RackMultipart20201030-4-6br838_html_30d1cab23f8251c1.png)

**Performance Evaluation**

After labeling the testing examples using the classifier, we can evaluate the algorithm&#39;s performance.

-We do so by calculating the percentage of correctly classified images from the testing set.

-Classification accuracy for certain images can vary significantly based on the feature extraction method used.

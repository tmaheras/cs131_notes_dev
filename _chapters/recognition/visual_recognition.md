---
title: Visual recognition
keywords: (insert comma-separated keywords here)
order: 12 # Lecture number for 2020
---
Table of Contents
-----------------

**12.1 Introduction to Object Recognition**
-------------------------------------------

-   Visual Recognition Tasks and Applications

-   Challenges

**12.2 k-Nearest Neighbors**
----------------------------

-   Machine Learning Framework

-   Nearest Neighbors Classifier

-   K-Nearest Neighbors

    -   Challenges

<!-- -->

-   Bias and Variance

    -   Bias Variance Tradeoff

-   No Free Lunch Theorem

**12.3 A Simple Object Recognition Pipeline**
---------------------------------------------

-   A Simple Image Classifier

-   Pipeline

-   Feature Extraction

-   Test Image Classification

-   Performance Evaluation

**12.1: Introduction to Object Recognition**
============================================

**<u>Visual Recognition Tasks and Applications</u>**

In general, the goal is to design an artificial visual system that can
mimic human vision capabilities.

Tasks

1.  **Classification.** Classify an entire image as belonging to a
    > certain category or not (e.g. does it contain a particular object
    > or not).

2.  **Detection.** Identifying if some object is in the image, and where
    > it is in the image. Essentially, this task is a combination of
    > classification and localization and can be more sophisticated by
    > detecting object semantic and geometric attributes such as
    > relative distances and sizes.

    -   Applications:

        -   Security

        -   Computational Photography

        -   Assistive Driving

3.  **Single Instance Recognition.** Recognizing whether a particular
    > instance of an object exists in the image and where in the image
    > is it located.

-   Applications:

    -   Landmark Detection and GPS

1.  **Activity or Event Recognition.** Being able to recognize what is
    > happening in an image and what activities are the subjects of the
    > image engaged in.

-   Applications:

    -   Human-Robot Interaction

    -   Sign Language

    -   Ambient Intelligence

    -   Sport Analysis

**<u>Challenges</u>**

There are many challenges one will encounter when building algorithms
that can classify human activities and events, detect and localize
objects, estimate semantic attributes and classify objects. These
challenges include:

1.  **A large number of categories for object classification.** For
    > example, ImageNet, a large dataset for object recognition,
    > contains over 14M images and over 20k categories

<img src="{{ site.baseurl }}/assets/images/media/image44.png" style="width:2.76827in;height:2.02261in" />

1.  **Viewpoint Variation:** The same image will appear to look quite
    > different depending on the angle at which it was taken. As shown
    > below, the geometric positions of the statue’s facial features
    > vary greatly depending on the position of the camera.

<img src="{{ site.baseurl }}/assets/images/media/image47.png" style="width:3.52747in;height:2.07851in" />

1.  **Illumination:** The position, intensity and color of the light
    > will impact how an object looks

<img src="{{ site.baseurl }}/assets/images/media/image46.png" style="width:3.22396in;height:1.26394in" />

1.  **Scale:** It is challenging to estimate and adjust for scale due to
    > the location of the camera, differences in size of other objects
    > in the image and instances of the same object category may vary
    > greatly in size

> <img src="{{ site.baseurl }}/assets/images/media/image34.png" style="width:1.72396in;height:2.30255in" />

1.  **Deformation:** An object may have certain characteristics that
    > make its body not rigid which may result in the object’s shape or
    > patterns differing between images. For example, a cat does not
    > have a rigid body and may take many different shapes

<img src="{{ site.baseurl }}/assets/images/media/image14.jpg" style="width:1.56609in;height:2.08812in" /><img src="{{ site.baseurl }}/assets/images/media/image39.png" style="width:1.62769in;height:2.0777in" />

1.  **Occlusion:** Objects are covered by other objects in an image and
    > are not fully visible

<img src="{{ site.baseurl }}/assets/images/media/image45.png" style="width:1.81652in;height:1.63021in" />

1.  **Background Clutter:** For objects that blend in with the
    > background, it is difficult to segment and accurately classify the
    > object

> <img src="{{ site.baseurl }}/assets/images/media/image42.png" style="width:2.33147in;height:1.65104in" />

1.  **Intra-Class Variation:** Instances within the same class can look
    > quite different from each other and the computer vision algorithms
    > must be able to account for this variation

<img src="{{ site.baseurl }}/assets/images/media/image43.png" style="width:2.30208in;height:1.43188in" />

 

**12.2: k-Nearest Neighbors**
=============================

**<u>Machine Learning Framework</u>**

<img src="{{ site.baseurl }}/assets/images/media/image29.png" style="width:2.34896in;height:1.44551in" />

**Training**: given a *training set* of labeled examples
[<img src="{{ site.baseurl }}/assets/images/media/image12.gif" style="width:1.34722in;height:0.16667in" />](https://latex-staging.easygenerator.com/eqneditor/editor.php?latex=%5C%7B(x_1%2Cy_1)%2C%20%E2%80%A6%2C(x_n%2Cy_n)%5C%7D#0)
estimate the prediction function
[<img src="{{ site.baseurl }}/assets/images/media/image2.png" style="height:0.15278in" />](https://www.codecogs.com/eqnedit.php?latex=f#0)
by minimizing the prediction error on the training set.

**Testing**: apply
[<img src="{{ site.baseurl }}/assets/images/media/image32.png" style="height:0.15278in" />](https://www.codecogs.com/eqnedit.php?latex=f#0)
to a never before seen test example
[<img src="{{ site.baseurl }}/assets/images/media/image18.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0)
and output predicted value
[<img src="{{ site.baseurl }}/assets/images/media/image33.png" style="width:0.61111in;height:0.16667in" />](https://www.codecogs.com/eqnedit.php?latex=y%20%3D%20f(x)#0).

**<u>Classification</u>**

Assign input vector to one of two or more classes

Any decision rule divides input space into *decision regions* separated
by *decision boundaries*.

Any input image can be mapped as a point in this space: we extract two
features in this case
[<img src="{{ site.baseurl }}/assets/images/media/image9.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_1#0)
and
[<img src="{{ site.baseurl }}/assets/images/media/image8.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_2#0)

<img src="{{ site.baseurl }}/assets/images/media/image13.png" style="width:3.18229in;height:2.33062in" />

**<u>Nearest Neighbor Classifier</u>**

Assign a label of nearest training data point to each test data point.

<img src="{{ site.baseurl }}/assets/images/media/image11.png" style="width:4.01563in;height:2.06573in" />

We compare the test instance with every set in the training set. We
chose an example that is closest in distance to the test image.

Above process can be seen as partitioning the feature space.

For example, if the input image is represented by two features
[<img src="{{ site.baseurl }}/assets/images/media/image16.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_1#0)
and
[<img src="{{ site.baseurl }}/assets/images/media/image20.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_2#0).
This process is similar to dividing this space into regions. So if the
red dots and black dots below are training samples, the boundaries
represent the area in the plane that is closest to each training sample.
When we see a new test sample we can map it to one of the boundaries:

<img src="{{ site.baseurl }}/assets/images/media/image24.png" style="width:3.04688in;height:2.76496in" />

This same idea works regardless of the dimensionality. The example below
is a 3D analogue of the 2D version above.

<img src="{{ site.baseurl }}/assets/images/media/image1.png" style="width:2.79688in;height:3.00405in" />

**<u>K-nearest neighbors</u>**

**Algorithm (training):**

1.  Store all training data points
    > [<img src="{{ site.baseurl }}/assets/images/media/image30.png" style="width:0.125in" />](https://www.codecogs.com/eqnedit.php?latex=x_i#0)
    > with their corresponding category labels
    > [<img src="{{ site.baseurl }}/assets/images/media/image22.png" style="width:0.11111in;height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=y_i#0)

**Algorithm (testing):**

1.  We are given a new test point
    > [<img src="{{ site.baseurl }}/assets/images/media/image17.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0).

2.  Compute distance to all training data points.

3.  Select
    > [<img src="{{ site.baseurl }}/assets/images/media/image23.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k#0)
    > training points closest to
    > [<img src="{{ site.baseurl }}/assets/images/media/image3.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0).

4.  Assign
    > [<img src="{{ site.baseurl }}/assets/images/media/image31.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0)
    > to label
    > [<img src="{{ site.baseurl }}/assets/images/media/image5.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=y#0)
    > that is most common among the
    > [<img src="{{ site.baseurl }}/assets/images/media/image7.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k#0)
    > nearest neighbors.

**Distance measurement** (one example)**:**

-   Euclidean distance:

> [<img src="{{ site.baseurl }}/assets/images/media/image38.png" style="width:2.04167in;height:0.61111in" />](https://www.codecogs.com/eqnedit.php?latex=d(x_n%2Cx_m)%20%3D%20%5Csqrt%7B%5Csum_%7Bj%3D1%7D%5E%7BD%7D%20(x_n%5Ej%20-%20x_m%5Ej)%7D#0)

Consider the following as training samples.

> <img src="{{ site.baseurl }}/assets/images/media/image4.png" style="width:2.86864in;height:2.39658in" />

So if
[<img src="{{ site.baseurl }}/assets/images/media/image10.png" style="width:0.375in;height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k%20%3D%201#0),
and black point is our test sample, we select the closest neighbor and
assign it a classification of red cross:

> <img src="{{ site.baseurl }}/assets/images/media/image21.png" style="width:3.07813in;height:2.70323in" />

Similarly, if
[<img src="{{ site.baseurl }}/assets/images/media/image19.png" style="width:0.375in;height:0.125in" />](https://www.codecogs.com/eqnedit.php?latex=k%20%3D%203#0),
we extract the 3 nearest neighbors. Since now we have green circles and
red crosses we select the classification with the most common label. In
this case there’s 3 green circles and 1 red cross so we assign a
classification of green circle this time.

> <img src="{{ site.baseurl }}/assets/images/media/image27.png" style="width:2.5096in;height:2.20246in" />

The decision boundaries produced by k-nearest neighbors are directly
established from the data. The algorithm is very flexible in capturing
complex relations between the training data points, and yet it is one of
the simplest classification algorithms. Because it is relatively simple,
it is a good option to try first when tackling a classification problem.
However, it is important to understand the drawbacks and challenges.

**k-Nearest Neighbors Challenges**

**<u>Choosing k</u>**

-   If k is too small, the decisions may be too sensitive to noisy data
    > points. For example, a training sample may have the wrong label.

-   If k is too large, the neighborhood may include points from other
    > classes.

Ex: Boundaries that can be obtained by changing the value of k from 1 to
15 on a dataset with 3 classes.

<img src="{{ site.baseurl }}/assets/images/media/image37.png" style="width:6.05729in;height:2.75684in" />

> Note that when k=1, the decision boundaries are highly complex.
> Whereas when k=15, the boundaries are simpler. This is usually good
> because it means they aren’t overly sensitive to noise in the data.

<img src="{{ site.baseurl }}/assets/images/media/image41.png" style="width:4.93229in;height:3.50001in" />

Decreasing k yields more complex decision boundaries, producing smaller
regions. More complex models yield smaller train error, but test error
tends to be higher. Increasing k yields simpler, smoother decision
boundaries (less complex model). We want to choose a k that minimizes
testing error.

So how do we choose k? The solution is **cross validation.**

-   For each value of k in the nearest neighbor algorithm:

    -   Create multiple train/test splits of the data. By creating
        > different sets of training and testing data, we create
        > multiple scenarios under which to validate our algorithm.

    -   Measure performance for each split.

    -   Average performance over all splits.

-   Select k with the best average performance.

<img src="{{ site.baseurl }}/assets/images/media/image40.png" style="width:4.94271in;height:2.46343in" />

**<u>Euclidean Measurement</u>**

When dimensionality of vectors is high, sometimes Euclidean distance
measurement can produce counterintuitive results.

Ex:

<img src="{{ site.baseurl }}/assets/images/media/image35.png" style="width:5.79688in;height:1.3749in" />

Solve this by normalizing vectors to be of unit length

**<u>Curse of Dimensionality</u>**

**Generalization:**

> <img src="{{ site.baseurl }}/assets/images/media/image15.png" style="width:4.7389in;height:2.88677in" />

1.  Given a training set of images for which I have labels that are
    > known...

2.  And a new test set for which to predict the labels...

3.  How well does the model I train on the training data generalizes
    > from that data into the test set?

> *What’s the error of my classifier in the testing set after I have
> trained in the training set?*
>
> **Takeaway:** Can one know beforehand which of the potential
> classifiers one can use, and will that classifier produce the lowest
> generalization error or lowest test error in the testing set.

**Bias and Variance:**

Two Components of Generalization Error:

-   **Bias:** *how much the average model over all training sets differ
    > from the true model?*

-   **Variance**: *how much models estimated from different training
    > sets differ from each other*

These result in:

-   **Underfitting**: *model is too “simple” to represent all the
    > relevant class characteristics*

    -   High bias and low variance

    -   High training error and high test error

-   **Overfitting:** *model is too “complex” and fits irrelevant
    > characteristics (noise) in the data*

    -   Low bias and high variance

    -   Low training error and high test error

To put these ideas into visual context, we will create a simplified
case:

Suppose we are fitting an equation to the red points in this plot:

-   The True equation is green line

    -   (To be estimated by the red points)

-   What models can best fit the true equation?

> <img src="{{ site.baseurl }}/assets/images/media/image28.png" style="width:4.21354in;height:2.68135in" />
>
> Linear model?

-   Too few parameters

    -   two coefficients in the equation

    -   Too simple to fit the red points

-   **Underfitting**

> *A model with too few parameters is INACCURATE because of a LARGE
> bias.*
>
> *It doesn’t have enough flexibility or capacity to fit the data.*
>
> <img src="{{ site.baseurl }}/assets/images/media/image6.png" style="width:4.14089in;height:2.54924in" />
>
> High degree polynomial model:

-   Too many parameters.

    -   Our model changes dramatically with each red point

-   **Overfitting**

<!-- -->

-   *A model with too MANY parameters is INACCURATE because of a large
    > variance.*

    -   *Too sensitive to the specific training set we are using, and
        > therefore, a very large variance.*

**Bias versus Variance Tradeoff**

Best explained by this plot:

<img src="{{ site.baseurl }}/assets/images/media/image26.png" style="width:5.63542in;height:3.57292in" />

> **Key observations:**

-   The more complex the model, the lower the Bias.

-   The more complex the model, the HIGHER the variance

-   There is a sweet spot between these two.

> We can do model selection to select the model complexity that will
> obtain the lowest total error. HOWEVER, how do we know what other
> classifiers have the lowest possible error?

**No Free Lunch Theorem:**

> Unfortunately we can’t really tell beforehand which classifier will
> have the lowest generalization error.
>
> We can only get generalization through some assumptions.

**3 Kinds of Errors:**

-   *Inherent:* unavoidable

-   *Bias:* due to over-simplifications

-   *Variance:* due to inability to perfectly estimate parameters from
    > limited data

**How to reduce these errors?:**

-   *Reduce Variance:*

    -   Choose a simpler classifier

    -   Regularize the parameters

    -   Get more training data

-   *Reduce Bias:*

    -   Choose a more complex classifier

    -   Increase the amount of features / parameters

    -   Get more training data

**Machine Learning Methods Final Tips:**

Know your data

-   How much supervision do you have?

-   How many training examples can you afford?

-   How noisy?

Know your goal:

-   Affects your choices of representation

-   Affects your choices of learning algorithms

-   Affects your choices of evaluation metrics

*Understand the math behind each machine learning algorithm you are
choosing!*

 

**12.3: A Simple Object Recognition Pipeline**
==============================================

**<u>A Simple Image Classifier</u>**

For this example of an object recognition pipeline, we will use a simple
image classifier.

> -We therefore will be using a prediction function that takes feature
> representations of images as inputs and yields classifications of
> their contents as outputs.

**Pipeline**

The pipeline contains the following steps:

1.  First, we ingest a collection of training images and extract image
    > features from them

2.  We train the classifier using the image features of the images as
    > well as their (ground-truth) training labels.

3.  After training this classifier, we can now use it to predict
    > classifications for unseen testing images:

    1.  First, we extract features from the test images

    2.  Then we feed those features to our trained classifier to output
        > the prediction for the classification of the object.

This pipeline is illustrated below:

<img src="{{ site.baseurl }}/assets/images/media/image25.png" style="width:4.49479in;height:2.41307in" />

**Feature Extraction**

Here are a few different methods for extracting these image features
from training and testing images:

-   Extracting color features, for example by creating a color histogram
    > to represent the frequency with which each color appears in the
    > image.

    -   These features are translation, scale, and rotation invariant

    -   However, they are not occlusion invariant

-   Using features that capture the object’s shape

    -   These features are also translation and rotation invariant, but
        > not occlusion invariant

-   Extracting features that capture *local* shape

    -   This can be done with keypoint detection, followed by extracting
        > the shape of each keypoint

    -   Such features are only translation and scale invariant

-   Using filter banks to extract texture-based features

    -   The resulting features are translation invariant

**Test Image Classification**

When training is complete and features have been extracted, the trained
classifier is then used to classify each new test image.

> -For example, if nearest-neighbor is used, the test image will be
> classified with the same label as its closest neighbor in the training
> set. In the example below, the test image would be classified as a
> blue square:
>
> <img src="{{ site.baseurl }}/assets/images/media/image36.png" style="width:3.67188in;height:1.54075in" />

**Performance Evaluation**

After labeling the testing examples using the classifier, we can
evaluate the algorithm’s performance.

> -We do so by calculating the percentage of correctly classified images
> from the testing set.
>
> -Classification accuracy for certain images can vary significantly
> based on the feature extraction method used.


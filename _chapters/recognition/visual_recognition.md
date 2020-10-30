---
title: Visual recognition
keywords: (insert comma-separated keywords here)
order: 12 # Lecture number for 2020
---
**Lecture 12: Object Recognition, kNN**
---------------------------------------

Table of Contents
-----------------

**12.1 Introduction to Object Recognition**
-------------------------------------------

> 12.1.1 Visual Recognition Tasks and Applications
>
> 12.1.2 Challenges

**12.2 k-Nearest Neighbors**
----------------------------

12.2.1 Machine Learning Framework

12.2.2 Classification

> 12.2.3 Nearest Neighbors Classifier
>
> 12.2.4 K-Nearest Neighbors Algorithm
>
> 12.2.5 K-Nearest Neighbors Challenges

-   Choosing k

-   Euclidean Measure

-   Curse of Dimensionality

> 12.2.5 Bias and Variance

-   Generalization

-   Bias and Variance

-   Bias Variance Tradeoff

-   No Free Lunch Theorem

> 12.2.6 Machine Learning Method Final Tips

**12.3 A Simple Object Recognition Pipeline**
---------------------------------------------

> 12.3.1 A Simple Image Classifier
>
> 12.3.2 Pipeline
>
> 12.3.3 Feature Extraction
>
> 12.3.4 Test Image Classification
>
> 12.3.5 Performance Evaluation

References:

-   CS 131 Lecture 12, Fall 2020

-   [<u>https://github.com/StanfordVL/cs131\_notes/blob/master/lecture11/lecture11.pdf</u>](https://github.com/StanfordVL/cs131_notes/blob/master/lecture11/lecture11.pdf)

**12.1: Introduction to Object Recognition**
============================================

**<u>12.1.1 Visual Recognition Tasks and Applications</u>**
===========================================================

In general, the goal is to design an artificial visual system that can
mimic human vision capabilities. The following are some tasks that can
be performed:

1.  **Classification.** Classify an entire image as belonging to a
    certain category or not (e.g. does it contain a particular object
    or not).

2.  **Detection.** Identifying if some object is in the image, and where
    it is in the image. Essentially, this task is a combination of
    classification and localization and can be more sophisticated by
    detecting object semantic and geometric attributes such as
    relative distances and sizes.

    -   Applications:

        -   Security

        -   Computational Photography

        -   Assistive Driving

3.  **Single Instance Recognition.** Recognizing whether a particular
    instance of an object exists in the image and where in the image
    is it located.

-   Applications:

    -   Landmark Detection and GPS

1.  **Activity or Event Recognition.** Being able to recognize what is
    happening in an image and what activities are the subjects of the
    image engaged in.

-   Applications:

    -   Human-Robot Interaction

    -   Sign Language

    -   Ambient Intelligence

    -   Sport Analysis

**<u>12.1.2 Challenges</u>**

There are many challenges one will encounter when building algorithms
that can classify human activities and events, detect and localize
objects, estimate semantic attributes and classify objects. These
challenges include:

1.  **A large number of categories for object classification.** For
    example, ImageNet, a large dataset for object recognition,
    contains over 14M images and over 20k categories

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image46.png" style="width:2.76827in;height:2.02261in" />

1.  **Viewpoint Variation:** The same image will appear to look quite
    different depending on the angle at which it was taken. As shown
    below, the geometric positions of the statue’s facial features
    vary greatly depending on the position of the camera.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image39.png" style="width:3.52747in;height:2.07851in" />

1.  **Illumination:** The position, intensity and color of the light
    will impact how an object looks

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image48.png" style="width:3.22396in;height:1.26394in" />

1.  **Scale:** It is challenging to estimate and adjust for scale due to
    the location of the camera, differences in size of other objects
    in the image and instances of the same object category may vary
    greatly in size

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image38.png" style="width:1.72396in;height:2.30255in" />

1.  **Deformation:** An object may have certain characteristics that
    make its body not rigid which may result in the object’s shape or
    patterns differing between images. For example, a cat does not
    have a rigid body and may take many different shapes

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image25.jpg" style="width:1.56609in;height:2.08812in" /><img src="{{ site.baseurl }}/assets/images/mediaFINAL/image42.png" style="width:1.62769in;height:2.0777in" />

1.  **Occlusion:** Objects are covered by other objects in an image and
    are not fully visible

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image45.png" style="width:1.81652in;height:1.63021in" />

1.  **Background Clutter:** For objects that blend in with the
    background, it is difficult to segment and accurately classify the
    object

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image44.png" style="width:2.33333in;height:2.76042in" />

1.  **Intra-Class Variation:** Instances within the same class can look
    quite different from each other and the computer vision algorithms
    must be able to account for this variation

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image43.png" style="width:2.30208in;height:1.43188in" />

**12.2: k-Nearest Neighbors**
=============================

**<u>12.2.1 Machine Learning Framework</u>**

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image18.png" style="width:2.34896in;height:1.44551in" />

**Training**: given a *training set* of labeled examples
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image1.gif" style="width:1.34722in;height:0.16667in" />](https://latex-staging.easygenerator.com/eqneditor/editor.php?latex=%5C%7B(x_1%2Cy_1)%2C%20%E2%80%A6%2C(x_n%2Cy_n)%5C%7D#0)
estimate the prediction function
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image10.png" style="height:0.15278in" />](https://www.codecogs.com/eqnedit.php?latex=f#0)
by minimizing the prediction error on the training set.

**Testing**: apply
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image31.png" style="height:0.15278in" />](https://www.codecogs.com/eqnedit.php?latex=f#0)
to a never before seen test example
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image5.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0)
and output predicted value
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image33.png" style="width:0.61111in;height:0.16667in" />](https://www.codecogs.com/eqnedit.php?latex=y%20%3D%20f(x)#0).

**<u>12.2.2 Classification</u>**

Assign input vector to one of two or more classes

Any decision rule divides input space into *decision regions* separated
by *decision boundaries*.

Any input image can be mapped as a point in this space: we extract two
features in this case
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image16.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_1#0)
and
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image24.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_2#0)

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image14.png" style="width:3.18229in;height:2.33062in" />

**<u>12.2.3 Nearest Neighbor Classifier</u>**

Assign a label of nearest training data point to each test data point.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image23.png" style="width:4.01563in;height:2.06573in" />

We compare the test instance with every set in the training set. We
chose an example that is closest in distance to the test image.

Above process can be seen as partitioning the feature space.

For example, if the input image is represented by two features
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image15.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_1#0)
and
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image7.png" style="width:0.13889in" />](https://www.codecogs.com/eqnedit.php?latex=x_2#0).
This process is similar to dividing this space into regions. So if the
red dots and black dots below are training samples, the boundaries
represent the area in the plane that is closest to each training sample.
When we see a new test sample we can map it to one of the boundaries:

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image2.png" style="width:3.04688in;height:2.76496in" />

This same idea works regardless of the dimensionality. The example below
is a 3D analogue of the 2D version above.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image19.png" style="width:2.79688in;height:3.00405in" />

**<u>12.2.4 k-Nearest Neighbors</u>**

**Algorithm (training):**

1.  Store all training data points
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image29.png" style="width:0.125in" />](https://www.codecogs.com/eqnedit.php?latex=x_i#0)
    with their corresponding category labels
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image32.png" style="width:0.11111in;height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=y_i#0)

**Algorithm (testing):**

1.  We are given a new test point
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image4.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0).

2.  Compute distance to all training data points.

3.  Select
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image8.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k#0)
    training points closest to
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image6.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0).

4.  Assign
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image21.png" />](https://www.codecogs.com/eqnedit.php?latex=x#0)
    to label
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image17.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=y#0)
    that is most common among the
    [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image12.png" style="height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k#0)
    nearest neighbors.

**Distance measurement** (one example)**:**

-   Euclidean distance:

> [<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image35.png" style="width:2.04167in;height:0.61111in" />](https://www.codecogs.com/eqnedit.php?latex=d(x_n%2Cx_m)%20%3D%20%5Csqrt%7B%5Csum_%7Bj%3D1%7D%5E%7BD%7D%20(x_n%5Ej%20-%20x_m%5Ej)%7D#0)

Consider the following as training samples.

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image11.png" style="width:2.86864in;height:2.39658in" />

So if
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image3.png" style="width:0.375in;height:0.11111in" />](https://www.codecogs.com/eqnedit.php?latex=k%20%3D%201#0),
and black point is our test sample, we select the closest neighbor and
assign it a classification of red cross:

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image20.png" style="width:3.07813in;height:2.70323in" />

Similarly, if
[<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image9.png" style="width:0.375in;height:0.125in" />](https://www.codecogs.com/eqnedit.php?latex=k%20%3D%203#0),
we extract the 3 nearest neighbors. Since now we have green circles and
red crosses we select the classification with the most common label. In
this case there’s 3 green circles and 1 red cross so we assign a
classification of green circle this time.

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image22.png" style="width:2.5096in;height:2.20246in" />

**<u>12.2.5 k-Nearest Neighbors Challenges:</u>**

kNN is one of the simplest classification algorithms, making it a good
option to try first when tackling a classification problem. Furthermore,
the decision boundaries produced by k-nearest neighbors are directly
established from the data. This results in the algorithm being very
flexible in capturing complex relations between the training data
points.

Now that we understand the benefits of kNN, let’s move onto the
challenges.

**<u>Choosing k</u>**

-   If k is too small, the decisions may be too sensitive to noisy data
    points. For example, a training sample may have the wrong label.

-   If k is too large, the neighborhood may include points from other
    classes.

Ex: Boundaries that can be obtained by changing the value of k from 1 to
15 on a dataset with 3 classes.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image40.png" style="width:6.05729in;height:2.75684in" />

> Note that when k=1, the decision boundaries are highly complex.
> Whereas when k=15, the boundaries are simpler. This is usually good
> because it means they aren’t overly sensitive to noise in the data.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image47.png" style="width:4.93229in;height:3.50001in" />

Decreasing k yields more complex decision boundaries, producing smaller
regions. More complex models yield smaller train error, but test error
tends to be higher. Increasing k yields simpler, smoother decision
boundaries (less complex model).

We want to choose a k that minimizes testing error.

**Solution: Cross Validation**

-   For each value of k in the nearest neighbor algorithm:

    -   Create multiple train/test splits of the data. By creating
        different sets of training and testing data, we create
        multiple scenarios under which to validate our algorithm.

    -   Measure performance for each split.

    -   Average performance over all splits.

-   Select k with the best average performance.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image41.png" style="width:4.94271in;height:2.46343in" />

**<u>Euclidean Measure</u>**

The Euclidean distance measurement can produce counterintuitive results.

Ex:

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image36.png" style="width:5.79688in;height:1.3749in" />

**Solution: Normalize the vectors to unit length.** Divide the vector by
its magnitude to get a magnitude of one.

**<u>Curse of Dimensionality</u>**

Assume 5,000 points are uniformly distributed in the unit hypercube
(cube in n dimensions whose side is equal to one), and we want to apply
5-NN. Suppose our query point is at the origin.

-   In 1 dimension, we must go a distance of 5/5000 = 0.001 on average
    to capture 5 nearest neighbors.

-   In 2 dimensions, we must go (0.001)<sup>1/2</sup> to get a square
    that contains 0.001 of the volume.

-   In d dimensions, we must go (0.001)<sup>1/d</sup>.

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image37.png" style="width:4.19271in;height:2.0426in" />

From this example, we see that the density of points goes down
exponentially as dimensionality increases. This means that points are
further apart and thus it is more difficult to find neighbors.

There are currently no good solutions to this problem.

**<u>12.2.5 Bias and Variance</u>**

**<u>Generalization:</u>**

There are many classification algorithms to choose from, and kNN may not
be the best option depending on your data. The best model to use is the
one that best generalizes from the data it was trained on to a new test
set.

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image13.png" style="width:4.7389in;height:2.88677in" />

We must ask, given a training set of images for which I have labels that
are known and a new test set for which to predict the labels, how well
does the model I train on the training data generalizes from that data
into the test set? *What’s the error of my classifier in the testing set
after I have trained in the training set?* Which classifier will produce
the lowest generalization error (the lowest testing error).

**<u>Bias and Variance:</u>**

Two Components of Generalization Error:

-   **Bias:** *how much the average model over all training sets differ
    from the true model?*

-   **Variance**: *how much models estimated from different training
    sets differ from each other*

These result in:

-   **Underfitting**: *model is too “simple” to represent all the
    relevant class characteristics*

    -   High bias and low variance

    -   High training error and high test error

-   **Overfitting:** *model is too “complex” and fits irrelevant
    characteristics (noise) in the data*

    -   Low bias and high variance

    -   Low training error and high test error

To put these ideas into visual context, we will create a simplified
case:

Suppose we are fitting an equation to the red points in this plot:

-   The True equation is green line

    -   (To be estimated by the red points)

-   What models can best fit the true equation?

> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image28.png" style="width:4.22396in;height:2.68039in" />

-   Linear model?

<!-- -->

-   Too few parameters

    -   Two coefficients in the equation

    -   Too simple to fit the red points

-   Underfitting

> *A model with too few parameters is INACCURATE because of a LARGE
> bias.*
>
> *It doesn’t have enough flexibility or capacity to fit the data.*
>
> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image26.png" style="width:4.14089in;height:2.54924in" />

-   High degree polynomial model:

    -   Too many parameters.

        -   Our model changes dramatically with each red point

    -   **Overfitting**

> *A model with too MANY parameters is INACCURATE because of a large
> variance. It is too sensitive to the specific training set we are
> using, and therefore has a very large variance.*

**<u>Bias versus Variance Tradeoff</u>**

Best explained by this plot:

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image27.png" style="width:5.63542in;height:3.57292in" />

**Key observations:**

-   The more complex the model, the lower the Bias.

-   The more complex the model, the HIGHER the variance

-   There is a sweet spot between these two.

We can do model selection to select the model complexity that will
obtain the lowest total error. HOWEVER, how do we know what other
classifiers have the lowest possible error?

**<u>No Free Lunch Theorem:</u>**

Unfortunately we can’t really tell beforehand which classifier will have
the lowest generalization error.

We can only get generalization through some assumptions.

**3 Kinds of Errors:**

-   *Inherent:* unavoidable

-   *Bias:* due to over-simplifications

-   *Variance:* due to inability to perfectly estimate parameters from
    limited data

**How to reduce these errors?:**

-   *Reduce Variance:*

    -   Choose a simpler classifier

    -   Regularize the parameters

    -   Get more training data

-   *Reduce Bias:*

    -   Choose a more complex classifier

    -   Increase the amount of features / parameters

    -   Get more training data

**<u>12.2.7 Machine Learning Methods Final Tips:</u>**

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

**<u>12.3.1 A Simple Image Classifier</u>**

For this example of an object recognition pipeline, we will use a simple
image classifier.

> -We therefore will be using a prediction function that takes feature
> representations of images as inputs and yields classifications of
> their contents as outputs.

**<u>12.3.2 Pipeline</u>**

The pipeline contains the following steps:

1.  First, we ingest a collection of training images and extract image
    features from them

2.  We train the classifier using the image features of the images as
    well as their (ground-truth) training labels.

3.  After training this classifier, we can now use it to predict
    classifications for unseen testing images:

    1.  First, we extract features from the test images

    2.  Then we feed those features to our trained classifier to output
        the prediction for the classification of the object.

This pipeline is illustrated below:

<img src="{{ site.baseurl }}/assets/images/mediaFINAL/image34.png" style="width:4.49479in;height:2.41307in" />

**<u>12.3.3 Feature Extraction</u>**

Here are a few different methods for extracting these image features
from training and testing images:

-   Extracting color features, for example by creating a color histogram
    to represent the frequency with which each color appears in the
    image.

    -   These features are translation, scale, and rotation invariant

    -   However, they are not occlusion invariant

-   Using features that capture the object’s shape

    -   These features are also translation and rotation invariant, but
        not occlusion invariant

-   Extracting features that capture *local* shape

    -   This can be done with keypoint detection, followed by extracting
        the shape of each keypoint

    -   Such features are only translation and scale invariant

-   Using filter banks to extract texture-based features

    -   The resulting features are translation invariant

**<u>12.3.4 Test Image Classification</u>**

When training is complete and features have been extracted, the trained
classifier is then used to classify each new test image.

> -For example, if nearest-neighbor is used, the test image will be
> classified with the same label as its closest neighbor in the training
> set. In the example below, the test image would be classified as a
> blue square:
>
> <img src="{{ site.baseurl }}/assets/images/mediaFINAL/image30.png" style="width:3.67188in;height:1.54075in" />

**<u>12.3.5 Performance Evaluation</u>**

After labeling the testing examples using the classifier, we can
evaluate the algorithm’s performance.

> -We do so by calculating the percentage of correctly classified images
> from the testing set.
>
> -Classification accuracy for certain images can vary significantly
> based on the feature extraction method used.



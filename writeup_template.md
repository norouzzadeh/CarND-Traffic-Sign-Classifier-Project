# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

# ** Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distributions.png  "Distributions"
[image2]: ./examples/random_images.png  "Random images"
[image3]: ./german_traffic_signs/09.jpg "No passing"
[image4]: ./german_traffic_signs/14.jpg "Stop"
[image5]: ./german_traffic_signs/17.jpg "No entry"
[image6]: ./german_traffic_signs/18.jpg "General caution"
[image7]: ./german_traffic_signs/25.jpg "Road work"
[image8]: ./examples/conv1.png  "Convolution1"
[image9]: ./examples/conv2.png  "Convolution2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/norouzzadeh/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The following image shows the distributions of the training, validation and test sets. As it can be seen, these sets have the same distrution.

![alt text][image1]

Furthermore, five random images are visualized here from the training set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize images. This makes the learning process much easier. The conversion to the gray scale was not necessary and the color images are used for the training. Also, the network is tested without any extra data augmentation. As the testing accuracy met the minimum requirements, it wasn't necessary to add more data to the data set. It is possible to reach more accuracy by adding augmented data to the data set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| FLATTEN				|												|
| Dropout				|												|
| Fully connected		| Input = 400, Output = 120						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Input = 120, Output = 84						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Input = 84, Output = 43						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with the batch size of 128, learning rate of 0.0005 and number of epochs equal to 25.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.954 
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture was the default LeNet architecture used for the MNIST example.

* What were some problems with the initial architecture?

The initial architecture had low learning capability as it was too simple to learn the traffic sign images.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The architecture was adjusted by adding some drop out layers. 

* Which parameters were tuned? How were they adjusted and why?

Later, the learning rate was adjusted. the learning rate of 0.001 was large and changing the learning rate to 0.0005 increased the validation accuracy. Furthermore, the keep probability was tuned where the value of 0.6 acheived good results.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

A convolution layer is suitable for this problem as it can find traffic signs in any location in the image space. Also, the dropout layers make the neural network less dependent to specific weights and increase both validation and test accuracies.


If a well known architecture was chosen:
* What architecture was chosen?

The LeNet architecture was used as the basic architecture for this problem.

* Why did you believe it would be relevant to the traffic sign application?

The basis LeNet architecture was able to classify numbers and adding some layers to this architecture makes it complex enough to classify traffic sign images.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
As the accuracies of the training, validation and test sets are above the 93%, it can be concluded that the final model is abale to classify traffic sign images.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

The traffic signs are slightly rotated in these images. Also, they have different illumination and background. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing   									| 
| Stop sign     		| Stop sign 									|
| No entry				| No entry										|
| General caution	    | General caution					 			|
| Road work				| Road work      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is pure sure that this is a no passing sign (probability of 0.98), and the image does contain a no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| No passing   									| 
| .009     				| Vehicles over 3.5 metric tons prohibited		|
| .002					| End of no passing								|
| .001	      			| No passing for vehicles over 3.5 metric tons 	|
| .001				    | Dangerous curve to the left       			|


For the second image, the model is pure sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop   										| 
| .009     				| Yield											|
| .002					| No entry										|
| .001	      			| Speed limit (30km/h)							|
| .001				    | Road work										|

For the third image, the model is pure sure that this is a no entry sign (probability of 0.99), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry   									| 
| .00001     			| Stop											|
| .0000000000008		| Bicycles crossing								|
| .0000000000004		| Priority road									|
| .0000000000004		| Traffic signals								|

For the fourth image, the model is pure sure that this is a general caution sign (probability of 0.99), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General caution								| 
| .0008     			| Pedestrians									|
| .000003				| Traffic signals 								|
| .00000002				| Road narrows on the right 					|
| .000000000001			| Right-of-way at the next intersection			|

For the fifth image, the model is pure sure that this is a road work sign (probability of 0.95), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Road work										| 
| .03     				| Wild animals crossing							|
| .004					| Double curve  								|
| .003					| Dangerous curve to the left  					|
| .002					| Bicycles crossing								|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The feature maps of the first convolution layer is shown in the following image:

![alt text][image8]

The following image shows the feature maps of the second convolution layer:

![alt text][image9]

As it can be seen in these images, the neural networks learns some features from the images. In the first layer, the features are more related to the structure of a traffic sign. In the second layer the features are specific to the network. 




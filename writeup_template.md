# Traffic Sign Recognition 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./submit_pics/visualization.jpg "Visualization"
[image2]: ./submit_pics/grayscale.png "Grayscaling"
[image3]: ./submit_pics/training_dataset.png "Trainning Dataset"
[image4]: ./submit_pics/normalized_image.png "Normalized Image"
[image5]: ./submit_pics/new_test_images.png "New Test Images"
[image6]: ./submit_pics/prediction_accuracy.png "Prediction Accuracy"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to calculate summary statistics of the traffic,
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing how the data ...

![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### Preprocess data

* Converting images to grayscale: According to remind in the project and published baseline module written by Pierre Sermanet and Yann LeCun, it should definitely use grascale against rgb images, according to my analysis, grayscaled image not has too much information as well as rgb images, it could make the trainning in neural network faster.

* Normalizing the data to the range (-1,1): I firstly normalized data with formula (data / 255. - .5), secondly use the formula ((data -128) / 128.0), the later formula has a good result, because rgb pixel value range is (0,255), (255 - 128) /128.0 = 1, (0 - 128) / 128.0 = -1. But the validation accuracy is not good as I expected. Eventually I find third formula (data/122.5) -1, it do improve the accuracy, for more details, we can find in [udacity forum](https://discussions.udacity.com/t/lenet-producing-only-a-75-accuracy/236031/2)

* Reshape the images: The original data shape is (32, 32, 3), I reshape this to (32, 32, 1), that will be the first convolutional layer.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Here is an example of traffic sign image after normalizing.

![alt text][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
|  Input         		| 32x32x1 grayscaled images   							| 
| Layer1: Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
|  RELU					|												| activation function |
|  Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Layer2: Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16		|
|  RELU	    | activation function |
|  Max pooling	  | 2x2 stride,  outputs 5x5x16 				|
| Flatten   | outputs 400 |
| Layer3: Fully connected		| input 400, output 120 									|
|  RELU					|												| activation function |
| Layer4: Fully connected		| input 120, output 84 									|
|  RELU					|												| activation function |
| Dropout |  									|
| Layer5: Fully connected		| input 84, output 43 									|
|  RELU					|												| activation function |
| Fully connected		| input 120, output 84 									|
| Dropout | keep_dropout_probability 0.7 									|

 
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I firstly shuffle the training data so that do not obtain entire minibatches of highly correlated datas, that helps training work well. And then I considered to split up the training dataset with sklearn's train_test_split, it returns a good accuracy result (>98.9%), but I eventually directly use the valid data from valid.p file, the accuracy is about 95%, but the valid data is come from the downlowded dataset, it's more reasonable to have a validation of training.

- epochs: 30
- batch size: 128
- learning rate: 0.0008
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.7


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 95.2% 
* test set accuracy of 92.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I choose Convolutional Neural Network(CNN) as the architecture, it mostly same architecture as the LeNet neural network, it has two convolutional layer and three fully connected layers. Besides, I use TensorFlow function to batch, scale and One-Hot encode the data. First, I runned with training and got a very low validation accuracy about 20%-40%, and also a very low test accuracy, it clearly indicated under fitting. I adjusted my pooling function from max pooling to average pooling, not worked, I learned that max pooling is decrease the size of the output and prevent over fitting NOT under fitting, actually here I need to concern is about under fitting. I adjusted preprocess, it worked. I also tried to split validation data from training data, it has good result in accuracy, but this will not use validation data downloaded from dataset. So I eventually use preprocess to the downloaded validata data. it has a better result, validation accuracy is more than 95%, but test accuracy only 60%-70%, that means over fitting, according to search on the internet, I eventually choose dropout units at 0.7 probablity, it works for reducing overfitting, the test accuracy is now more than 92%.


If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here I choosed are five German traffic signs that I found on the web:

![alt text][image5]

According to pictures pixel quanlity, the prediction is different, I choosed one sign image it photoed in the night, lable is 35(go straight), the prediction only get 98%, the matched traffic sign image is lable 37(go straight or Right), it's very similar but not same.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or Left      		| Go straight or Left   									| 
| Priority Road    			| Priority Road 										|
| Yield					| Yield											|
| Go Straight	      		| Go Straight					 				|
| 60km/h Speed Limits			| NOT SURE						|
| 30km/h Speed Limits| 50km/h Speed Limits |
| Keep Right | Keep Right |
| No Entry | No Entry |


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This has the same result of Test set accuracy, I runned several times to prediction new images, but I am sure the prediction more higher if new images more clear.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, send image, third image, seventh image, eigth image, the model is very sure match the sign, and probability is 1.0. The sixth image is relatively sure that this si 30km/h speed limits sign, different sess with different prediction result, it can be predicted most time, the 2nd prediction probability 0.12, but that is the correct image. For the fifth image, it has probability of 0.48, the top prediction is wrong, but the 2nd prediction is right image. For the fourth image, it has a correct prediction with probability of 0.82.

The top eight soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00 | Go straight or Left   									| 
| 1.00    			| Priority Road 										|
| 1.00					| Yield											|
| 0.82	      		| Go Straight					 				|
| 0.48			|     |
| 0.88 | 50km/h Speed Limits |
| 1.0 | Keep Right |
| 1.0 | No Entry |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
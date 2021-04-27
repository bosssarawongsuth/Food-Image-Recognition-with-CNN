# GA DSi Capstone Project: Food Image Recognition using Convolutional Neural Network
## Jetnipat Sarawongsuth 
### Problem Statement

Connectify.ai is a Singaporean tech start up that focuses on providing AI solutions to the healthcare industry. One of the challenges that they are trying to tackle is developing a mobile application that helps monitor the food intakes for people with diabetes. Part of the monitoring and recommendation process includes food journalling which involves the users entering the food eaten at each meal during the day. This manual process can become cumbersome and thus alternatives such as having the users take pictures of their food and automatically identifying the name of the food is being considered. I will be developing image classification models using Convolutional Neural Networks that can be used to identify the name of some of the more popular food in Asia. This can eventually be integrated into the mobile application to make the food journalling process more user friendly.


### Dataset
#### Training & Validation Sets
The training and validation food images are taken from a well known Food 101 dataset which can be downloaded from [Kaggle](https://www.kaggle.com/dansbecker/food-101). The original dataset contains 101 categories of food with 1000 images in each category. However, since this application will be mostly focused in SEA, we decided that at this stage, we only be using the food category that are more popular in SEA. This reduced the number of categories to 29 resulting in 29000 food images in total (before image augmentation).

#### Testing Set
The images in the testing set were webscraped from search engines such as Google and Bing. This testing set includes 580 images in total, each with 20 images.


***These images across the training, validation and testing sets are above 256x256 in resolution.***

<img src="https://github.com/bosssarawongsuth/cnn_food_classification/blob/main/images/images.PNG?raw=true" style="float: left; margin: 20px; height: 55px">

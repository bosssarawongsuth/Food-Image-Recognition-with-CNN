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



### Evaluation
<img src="https://github.com/bosssarawongsuth/cnn_food_classification/blob/main/images/evaluation.png?raw=true" style="float: left; margin: 20px; height: 55px">

The graph above shows the training, validation and testing accuracy for the 8 models trained. They are ranked on the performance on the unseen testing set. In this case, 'Inception V3 Dropout', 'InceptionV3 GAP', 'Inception-ResNetV2' and 'Inception-ResNetV2 Dropout' all scored 0.84 accuracy on the testing set. Thus, these four models generally have low bias. However, as the graph shows, when we take into account the training and validation accuracies, we can see that 'Inception V3 Dropout' generalises the best without suffering from the overfitting problem (high variance). Therefore, this model will be used as the classification engine for our web application.



### Conclusion

In conclusion, the best model created (Inception V3 Dropout) was able to satisfactorily classify the 29 different classes of food. Eventhough some food classes were shown to be more challenging than others, the model was still able to perform far better than the baseline accuracy of 3.4%. The model could be deployed on AWS service such as SageMaker to make the model accessible by mobile applications.

However, the model does not come without any limitations. The first limitation is the handling of images with multiple food classes in them. This arises because some food classes often come with another. For example, Fish and Chips vs French Fries. A possible solution to this problem might be to assign class weight when training to priotise certain classes over others. The second limitation is the problem with large intra-class diversity. This is where food images belonging to the same class might look drastically different due to different presentations, etc. This could be overcome by acquiring more training data and/or split classes into subclasses. The third limitation is having large inter-class similarity. This issue arises when there are multiple food classes that look similar. For example, Sushi vs Sashimi. A possible solution for this would be the acquire more training data and/or group very similar classes together into one umbrella class.

The model currently only predicts 29 food classes. This is something that future iterations could improve by introducing more classes of food. This would require a substancial amount of labelled food images so the model could be refitted.

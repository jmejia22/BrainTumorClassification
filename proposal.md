# Brain Tumor Classification
Using Deep Learning to classify whether an MRI image contains a tumor.


## Project Proposal
### 1. Abstract
For our PIC 16B project, we hope to create a machine learning pipeline in which we classify brain tumors via MRI image data. We hope to create a model that can analyze a given MRI image and classify whether the brain is healthy or contains abnormalities. We would like to build a project that can help others and also offer a second “opinion” to doctors/researchers when they are interpreting the results of an MRI. In order to achieve these goals, we plan to use deep learning, more specifically, implement a convolutional neural network in order to classify MRI images as healthy or abnormal. 


### 2. Planned Deliverables
In our eyes, we consider “full success” to be creating a website in which users can upload MRI images to our website and be given a response of “healthy” or “brain abnormality present”. In addition, if an abnormality is found, then we would like to draw a red box around the region of the predicted brain abnormality. This website would be especially beneficial to users as they could get an image classified instantaneously and also see exactly where brain tumors are present should our model detect one. 
On the other hand, “half success” in our eyes would be having a code repository on Github that demonstrates the machine learning pipeline we created which could be used by others should they want to use or build upon our algorithm. 
We understand the difficulties that may be present with building the web application tool indicated in our “full success” goal. Although this is our ultimate goal, we would also be more than proud of having a code repository that demonstrates our ML pipeline. 



### 3. Resources Required
The data that is needed for our project can be collected from the Kaggle website. We have found some datasets that we feel are more than sufficient in helping us achieve the goal of our proposal. This dataset contains 4 classes of MRI images (Giloma tumor, Meningioma tumor, Pituitary tumor, and no tumor). On the other hand, this dataset is classified as simply yes (brain tumor present) or no (healthy). In the first phase of our project, we will do preliminary analysis on both datasets to see which we think is better for our ambitions. 


### 4. Tools/Skills Required
In order to predict regions of brain cancer from an MRI image, we will need to create a machine learning model and train it on a dataset that includes both images of brain cancer and a normal brain. One machine learning model that we have in mind and believe would be appropriate for our task includes the use of convolution neural networks. In Sumit Saha’s article, called “[A Comprehensive Guide to  Convolutional Networks -- the EL15 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53),” Saha describes a convolutional neural network as a “Deep Learning algorithm which can take an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.” Given that we want to build a convolutional neural network, there are three Python libraries that we are aware of that can achieve this goal, namely [`keras`](https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python), [`tensorflow`](https://www.tensorflow.org/tutorials/images/classification), and/or [`Pytorch`](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). Once we build our model, we most likely would have to use `matplotlib` in order to outline the regions of cancer given an MRI image of a brain. 

### 5. Risks
One risk that could potentially stop us from achieving the full deliverable described above includes implementing an accurate and reliable machine learning model since we both are new to neural networks and the various Python libraries used to create them. Additionally, when building our website, we may have difficulty creating a feature that allows for import of good quality images for our machine learning model to process. 

### 6. Ethics
For our project, we suspect that biases may arise from the dataset that we choose to train our model with. For example, if we selected a dataset that only included MRI images of brains from people who were assigned male at birth, then our model may pick up on the features of a male brain that may not be predictive of all human brains. Then, when using the same model to predict regions of cancer on a female brain, the model may perform with less accuracy. This is an issue that needs to be addressed because in the real world if our model was used by surgeons to section out brain tumors, then females would have a higher risk of brain cancer recurrence as a result of measurement bias in our data. The same principle applies to race and ethnicity. Historically, access to healthcare is an issue for many disadvantaged groups so perhaps MRI imaging of the brain is not available to a large population. Thus, the dataset may contain more images from one race than another, which again can lead to a biased model. To address this issue, we will search for and train our model on a dataset that includes a diverse range of brain MRI images on the premise of race, physical or mental disability, and sex.

### 7. Tentative Timeline
- Week 5:
	- create our convolution neural network and train on data
- Week 7:
	- refine our model to achieve the best accuracy
		- may include testing on new datasets
	- write some of the expository text for our project
- Week 9:
	- create the website
	- finalize all project components 
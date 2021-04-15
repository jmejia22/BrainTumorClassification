# Brain Tumor Classification
Using Deep Learning to classify whether an MRI image contains a tumor.

```python
X . . 
. . . 
. O . 
```

## Project Proposal
### 1. Abstract


### 2. Planned Deliverables


### 3. Resources Required


### 4. Tools/Skills Required
In order to predict regions of brain cancer from an MRI image, we will need to create a machine learning model and train it on a dataset that includes both images of brain cancer and a normal brain. One machine learning model that we have in mind and believe would be appropriate for our task includes the use of convolution neural networks. In Sumit Saha’s article, called “[A Comprehensive Guide to  Convolutional Networks -- the EL15 way],” Saha describes a convolutional neural network as a “Deep Learning algorithm which can take an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other(https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).” Given that we want to build a convolutional neural network, there are three Python libraries that we are aware of that can achieve this goal, namely [`keras`](https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python), [`tensorflow`](https://www.tensorflow.org/tutorials/images/classification), and/or [`Pytorch`](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). Once we build our model, we most likely would have to use `matplotlib` in order to outline the regions of cancer given an MRI image of a brain. 

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
# Brain Tumor Classification
Using Deep Learning to classify whether an MRI image contains a tumor.

## Group Contribution Statement
Jacob: In terms of the code, I was responsible for writing the image augmentation function, preprocessing the data, helping build the CNN model, writing functions responsible for plotting images, predicting tumor position using linear regression, and adding functions to py files. I was also in charge of writing docstrings for about a quarter of the functions and adding comments throughout.  Wrote the explanatory text about machine learning bias. 

Wendy: I was responsible for finding the brain MRI data and doing the initial download/preprocessing of the images as well as exploratory analysis. I assisted in acquiring and loading the tumor mask data as well as configuring it into lists. I was also in charge of plotting the brain tumor images with their respective tumor masks. Additionally, I wrote the explanatory text on the evaluation of our model, linear regression model, image segmentation, and conclusion. I also cleaned up our repository and wrote the tutorial. 

## Tutorial

In this tutorial, we will demonstrate the functionality of our project through a Google Colab notebook, `brain_tumor_classification.ipynb`. We decided to use a Google Colab notebook as opposed to a jupyter notebook because Google Colab allows users to change their hardware accelerator to a GPU. Since we are building an image classification model, the GPU assists in speeding up runtime when it comes to training our model. Thus, we recommend using Google Colab and using their readily available (and free) GPU to run `brain_tumor_classification.ipynb`. To switch to Google Colab's GPU, click "Edit" &#8594; "Notebook settings" &#8594; "Hardware accelerator" &#8594; "GPU" &#8594; "Save". 

<p align="center">

<img src="/tutorial_images/GPU.png" width="372" height="265">

</p>


The next step would be to load in the necessary files onto your own Google Drive. Specifically, you need to upload the `data` folder as well as the three `.py` files, located in the `py` folder of this repository, into Colab Notebooks. Within the `data` folder, there are two additional folders, `tumor_data` and `no_tumor`. `tumor_data` contains the .mat files downloaded from [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). `no_tumor` data was downloaded from [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri). These two folders contain the brain MRI images of patients with tumors and patients without. 

<p align="center">

<img src="/tutorial_images/GoogleColab.png" width="534.5" height="237">

</p>

Once all of these files are located in your own Colab Notebook folder, everything should be set to run `brain_tumor_classsification.ipynb`. Just remember to switch to Google Colab's GPU when running the notebook! There is more explanatory text within the notebook that explains our process in preparing the data as well as building our model. 
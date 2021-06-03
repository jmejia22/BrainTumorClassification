# Brain Tumor Classification
Using Deep Learning to classify whether an MRI image contains a tumor.


## Tutorial

In this tutorial, we will demonstrate the functionality of our project through a Google Colab notebook, `brain_tumor_classification.ipynb`. We decided to use a Google Colab notebook as opposed to a jupyter notebook because Google Colab allows users to change their hardware accelerator to a GPU. Since we are building an image classification model, the GPU assists in speeding up runtime when it comes to training our model. Thus, we recommend using Google Colab and using their readily available (and free) GPU to run `brain_tumor_classification.ipynb`. To switch to Google Colab's GPU, click "Edit" &#8594; "Notebook settings" &#8594; "Hardware accelerator" &#8594; "GPU" &#8594; "Save". 

<p align="center">

<img src="/tutorial_images/GPU.png" width="372" height="265">

</p>


The next step would be to load in the necessary files onto your own Google Drive. Specifically, the next step would be to upload the `data` folder as well as the three `.py` files, located in the `py` folder, into Colab Notebooks. Within the `data` folder, there are two additional folders, `tumor_data` and `no_tumor`. `tumor_data` contains the .mat files downloaded from [figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427). `no_tumor` data was downloaded from [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).

<p align="center">

<img src="/tutorial_images/GoogleColab.png" width="534.5" height="237">

</p>

Once all of these files are located in your own Colab Notebook folder, everything should be set to run `brain_tumor_classsification.ipynb`. There is more explanatory text within this Google Colab notebook that explains our process in preparing the data as well as building our model. 
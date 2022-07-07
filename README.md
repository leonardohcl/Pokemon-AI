# Pokemon-AI

A quick example that uses Pokemon and their types to teach how to create an image classifier with python's libraries. The idea is to go from simpler algorithms to CNNs.

1. At [00_basic_classifier](https://github.com/leonardohcl/Pokemon-AI/blob/main/00_basic_classifier.py) there's a simple exemple using the histogram as a feature vector
2. At [01_create_data_augmentation](https://github.com/leonardohcl/Pokemon-AI/blob/main/01_create_data_augmentation.py) there's a code to create some data augmentation to the original images
3. At [02_improved_classifier](https://github.com/leonardohcl/Pokemon-AI/blob/main/02_improved_classifier.py) is an improvement to the first code, since there is now more data and some treatment for feature selection
4. At [03_training_a_cnn](https://github.com/leonardohcl/Pokemon-AI/blob/main/03_training_a_cnn.py) there's a basic example on how to train a CNN for the given scenario
5. At [04_test_the_trained_cnn](https://github.com/leonardohcl/Pokemon-AI/blob/main/04_test_the_trained_cnn.py) there's an exemple on how to use the trained CNN to predict images as well as on how to interpret the results using CAMs (Class Activation Maps)

All the scripts use helpers, but mostly from the [Pokemon](https://github.com/leonardohcl/Pokemon-AI/blob/main/Pokemon.py) module, where most of the dirty work is done. 

Also, the codes here are powered by a lot of libraries and packages. They are:

[SciKit](https://scikit-learn.org) v1.1

[PyTorch](https://pytorch.org/) v1.10  

[Pillow](https://pillow.readthedocs.io/en/stable/) v8.2

[Numpy](https://numpy.org/) v1.20.2
  
[pandas](https://pandas.pydata.org/) v1.2.4
  
[Matplotlib](https://matplotlib.org/) v3.4.3
  
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) v1.3.1

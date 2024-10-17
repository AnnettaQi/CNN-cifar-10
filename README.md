The repository is about: 

Use CNN to classify pictures into categories for cifar 10 dataset!

Data description:

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The goal of this project is about:

Experiment with fully connected neural networks and convolutional neural networks, using the Keras open source package. Keras is one of the simplest deep learning packages that serves as a wrapper on top of TensorFlow. Preliminary steps:

Familiarize yourself with Keras. Click on "Guides" and read the first two guides: "The functional API" and "The Sequential Model".
Download and install Keras on a machine with a GPU or use Google's Colaboratory environment, which allows you to run Keras code on a GPU in the cloud. Colab already has Keras pre-installed. To enable GPU acceleration, click on "edit", then "notebook settings" and select "GPU" for hardware acceleration. It is also possible to select "TPU", but the Keras code provided with this assignment will need to be modified in a non-trivial way to take advantage of TPU acceleration.
Download the base code for this assignment: cs480_fall20_asst4_cnn_cifar10.ipynb.
Answer the following questions by modifying the base code in cs480_fall20_asst4_cnn_cifar10.ipynb. Submit the modified Jupyter notebook via LEARN.

Part 1 : 

Compare the accuracy of the convolutional neural network in the file cs480_fall20_asst4_cnn_cifar10.ipynb on the cifar10 dataset to the accuracy of simple dense neural networks with 0, 1, 2, 3 and 4 hidden layers of 512 rectified linear units each.  
Modify the code in cs480_fall20_asst4_cnn_cifar10.ipynb to obtain simple dense neural networks with 0, 1, 2, 3 and 4 hidden layers of 512 rectified linear units (with a dropout rate of 0.5). 
Produce two graphs that contain 6 curves (one for the convolutional neural net and one for each dense neural net of 0-4 hidden layers). The y-axis is the accuracy and the x-axis is the number of epochs (\# of passes through the training set). Since neural networks take a while to train, cross-validation is not practical. Instead, produce one graph where all the curves correspond to the training accuracy and a second graph where all the curves correspond to the validation accuracy. Train the neural networks for 20 epochs. Although 20 epochs is not sufficient to reach convergence, it is sufficient to see the trend. Among the models abtained after each epoch, save the model that achieves the best validation accuracy and report its test accuracy. Save the following results in your Jupyter notebook:
The two graphs for training and validation accuracy.
For each architecture, print the test accuracy of the model that achieved the best validation accuracy among all epochs (i.e., one best test accuracy per network architecture).
Add some text to the Jupyter notebook to explain the results (i.e., why some models perform better or worse than other models).


Part 2 : 

Compare the accuracy achieved by rectified linear units and sigmoid units in the convolutional neural network in cs480_fall20_asst4_cnn_cifar10.ipynb. Modify the code in cs480_fall20_asst4_cnn_cifar10.ipynb to use sigmoid units. Produce two graphs (one for training accuracy and one for validation accuracy) that each contain 2 curves (one for rectified linear units and another one for sigmoid units). The y-axis is the accuracy and the x-axis is the number of epochs. Train the neural networks for 20 epochs. Although 20 epochs is not sufficient to reach convergence, it is sufficient to see the trend. Save the following results in your Jupyter notebook:
The two graphs for training and validation accuracy.
For each activation function, print the test accuracy of the model that achieved the best validation accuracy among all epochs (i.e., one best test accuracy per activation function).
Add some text to the Jupyter notebook to explain the results (i.e., why one model performs better or worse than the other model).


Part 3: 

Compare the accuracy achieved with and without drop out as well as with and without data augmentation in the convolutional neural network in cs480_fall20_asst4_cnn_cifar10.ipynb. Modify the code in cs480_fall20_asst4_cnn_cifar10.ipynb to turn on and off dropout as well as data augmentation. 
Produce two graphs (one for training accuracy and the other one for validation accuracy) that each contain 4 curves (dropout with data augmentation, dropout with no data augmentation, no dropout with data augmentation, no dropout with no data augmentation). The y-axis is the accuracy and the x-axis is the number of epochs.
Produce curves for as many epochs as you can up to 100 epochs.
For each combination of dropout and data augmentation, print the test accuracy of the model that achieved the best validation accuracy among all epochs (i.e., one best test accuracy per combination of dropout and data augmentation).
Add some text to the Jupyter notebook to explain the results (i.e., why did some models perform better or worse than other models and are the results consistent with the theory).


Part 4 : 

Compare the accuracy achieved when training the convolutional neural network in cs480_fall20_asst4_cnn_cifar10.ipynb with three different optimizers: RMSprop, Adagrad and Adam. Modify the code in cs480_fall20_asst4_cnn_cifar10.ipynb to use the Adagrad and Adam optimizers (with default parameters). Produce two graphs (one for training accuracy and the other one for validation accuracy) that each contain 3 curves (for RMSprop, Adagrad and Adam). The y-axis is the accuracy and the x-axis is the number of epochs. Produce curves for as many epochs as you can up to 100 epochs.
The two graphs for training and validation accuracy.
For each optimizer pringt the test accuracy of the model that achieved the best validation accuracy among all epochs (i.e., one best test accuracy per optimizer).
Add some text to the Jupyter notebook to explain the results (i.e., why did some optimizers perform better or worse than other optimizers).


Part 5 : 

Compare the accuracy of the convolutional neural network in cs480_fall20_asst4_cnn_cifar10.ipynb with a modified version that replaces each stack of (CONV2D, Activation, CONV2D, Activation) layers with 3x3 filters by a smaller stack of (CONV2D, Activation) layers with 5x5 filters. Produce two graphs (one for training accuracy and the other one for validation accuracy) that each contain 2 curves (for 3x3 filters and 5x5 filters). The y-axis is the accuracy and the x-axis is the number of epochs. Produce curves for as many epochs as you can up to 100 epochs.
The two graphs for training and validation accuracy.
For each filter configuration, print the test accuracy of the model that achieved the best validation accuracy among all epochs (i.e., one best test accuracy per filter configuration).
Add some text to the Jupyter notebook to explain the results (i.e., why did one architecture perform better or worse than the other architecture).

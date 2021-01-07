'''
	Author: Abhijit Adhikary
	Email: u7035746@anu.edu.au
'''

Required Libraries and Packages:
----------------------------------------------------
	numpy
	os
	torch (PyTorch)
	torchvision
	tensorboard (For visualization)
	cv2 (openCV) (For reading in the image data)
	matplotlib
	tqdm
	datetime


Running instructions:
----------------------------------------------------------------------------
(1) To run the network please run the 'run.py' file.


(2) Place the 'Subset For Assignment SFEW' folder containing the images in the 
	same directory to separate train, test and validation data. 

(3) 'helper_functions.py' file contains the necessary functions 
	and the 'train_test.py' file contains the train, test and validation functions.

(4) In the absense of a high performance GPU please reduce batch size of the
	dataloader.

(5) By default the model is set to training mode with 1 epoch.

(6) To visualize pretrained parameters enter in the commandline:
tensorboard --logdir=runs_pretrained
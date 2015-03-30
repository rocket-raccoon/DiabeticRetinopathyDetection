#Dependencies
import os

#This is the size the original images will be resized to
IMAGE_SIZE = (512, 512, 3)

#Booleans that determine what preprocessing steps to take
GREY = True             #This will greyscale images during preprocessing
CENTER_AND_SCALE = True #This will subtract mean from image pixels

#Sets the directories for the training images
CWD = os.getcwd()
ORIGINAL_TRAIN_DIR = CWD + "/train"
RESIZED_TRAIN_DIR = CWD + "/resized_train"
PROCESSED_TRAIN_DIR = CWD + "/processed_train"
GREY_TRAIN_DIR = CWD + "/grey_train"
TRAIN_LABELS = CWD + "/trainLabels.csv"

#Pickling settings
MINI_BATCH_SIZE = 40
TRAIN_PERC = 0.50

#Image properties
PIXELS_PER_IMAGE = IMAGE_SIZE[0]*IMAGE_SIZE[1]#*IMAGE_SIZE[2]

#Saved parameter settings
PARAMS_DIR = CWD + "/params"
PIXEL_MEANS_FILE = PARAMS_DIR + "/col_means.npy"




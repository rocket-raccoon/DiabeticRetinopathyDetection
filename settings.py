#Dependencies
import os

#This is the size the original images will be resized to
IMAGE_SIZE = (512, 512, 3)

#Booleans that determing what preprocessing steps to take
GREY = True
CENTER_AND_SCALE = False

#Sets the directories for the training images
CWD = os.getcwd()
ORIGINAL_TRAIN_DIR = CWD + "/train"
RESIZED_TRAIN_DIR = CWD + "/resized_train"
PROCESSED_TRAIN_DIR = CWD + "/processed_train"
TRAIN_LABELS = CWD + "/trainLabels.csv"
PICKLED_SETS_DIR = CWD + "/pickled_sets"

#Pickling settings
MINI_BATCH_SIZE = 100
TRAIN_PERC = 0.50

#Image properties
PIXELS_PER_IMAGE = 512*512



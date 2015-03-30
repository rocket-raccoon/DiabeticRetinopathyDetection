##################################################################
#The images in the Kaggle DR competition are way too large (>1MB)
#This script goes through each image and does two things:
# 1)  Trim the black around the eye as much as possible
# 2)  Resize the image from there to a more manageable size
##################################################################

#Dependencies
import os
import shutil
import math
import numpy as np
import settings
from scipy import misc
from Point import Point

LEFT = 0
RIGHT = 1
BOTTOM = 2
TOP = 3

#Crops out as much black from the edges of the image
#For each side (north, south, west, east) we start at the midpoint
#Move towards the center pixel at a time until we reach a point that isn't black
#Record these points and crop the image there
def crop_black(img):

    #Keep moving towards the center until you are no longer at a black pixel
    #Return the point reached
    def move_to_center_until_not_black(img, p, direction):
        while(True):
            if direction == LEFT:
                p.move_right()
            elif direction == RIGHT:
                p.move_left()
            elif direction == BOTTOM:
                p.move_up()
            elif direction == TOP:
                p.move_down()
            if any(img[p.x, p.y] > 3):
                break
        return p

    #Get left side crop point
    left = Point(img.shape[0]/2, 0)
    left = move_to_center_until_not_black(img, left, LEFT)

    #Get the right side crop point
    right = Point(img.shape[0]/2, img.shape[1])
    right = move_to_center_until_not_black(img, right, RIGHT)

    #Get the bottom side crop point
    bottom = Point(img.shape[0], img.shape[1]/2)
    bottom = move_to_center_until_not_black(img, bottom, BOTTOM)

    #Get the top side crop point
    top = Point(0, img.shape[1]/2)
    top = move_to_center_until_not_black(img, top, TOP)

    return img[top.x:bottom.x, left.y:right.y]

#Resize the image while preserving aspect ratio
#Pad with black pixels afterwards to form square
def resize_image(img, size):
    maxUnit = max(img.shape)
    h = int(math.floor(size[0]*(img.shape[0]/float(maxUnit))))
    w = int(math.floor(size[1]*(img.shape[1]/float(maxUnit))))
    img = misc.imresize(img, (h, w, 3))
    img_copy = np.zeros([size[0], size[1], 3])
    img_copy[0:img.shape[0], 0:img.shape[1], 0:3] = img
    return img_copy

#Grab all the image names
image_names = [i for i in os.listdir(settings.ORIGINAL_TRAIN_DIR) if ".jpeg" in i]

#Create a new folder to place the new images in if it doesn't exist already
if not os.path.isdir(settings.RESIZED_TRAIN_DIR):
    os.makedirs(settings.RESIZED_TRAIN_DIR)

for image_name in image_names:

    #Load the current image 
    image = misc.imread(settings.ORIGINAL_TRAIN_DIR + "/" + image_name)

    #Try to crop the black from the edges of the image, otherwise leave alone
    try:
        image = crop_black(image)
    except:
        pass
    image = resize_image(image, settings.IMAGE_SIZE)

    #Save the new image
    misc.imsave(settings.RESIZED_TRAIN_DIR + "/" + image_name, image)









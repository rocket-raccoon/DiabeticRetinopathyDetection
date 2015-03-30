######################################################
# This script takes all the images and does some     #
# basic preprocessing so that the deep learning goes #
# more smoothly later.  This includes (optionally):  #
# 1)  Mean subtraction                               #
# 2)  Greyscaling                                    #
# 3)  normalization of values between [0,1]          #
# 4)  Pickling data                                  #
# 5)  Making the minibatches have equal class values #
######################################################

#Dependencies
import os
import scipy.misc
import numpy as np
import settings
import csv
import cPickle
from collections import Counter

#Clears a folder of all contents (not recursively though)
def clear_folder(folder):
    for content in os.listdir(folder):
        if os.path.isfile(folder + "/" + content):
            os.unlink(folder + "/" + content)

#Turns the image matrix into greyscale
def rgb2gray(image):
    return np.dot(image[:,:,:3], [0.299, 0.587, 0.144])

#Load the training labels as a dictionary of (image_name, label) pairs
def load_labels(labels_file_path):
    with open(labels_file_path, 'rb') as csvfile:
        label_reader = csv.reader(csvfile, delimiter=",")
        next(label_reader, None)
        labels_dict = {name: int(label) for (name, label) in label_reader}
    return labels_dict

#Create a training and test set from a small subset of indices
def create_dataset(image_names, labels_dict, pixel_means, image_dir):
    images = []
    labels = []
    for image_name in image_names:
        image = scipy.misc.imread(image_dir + "/" + image_name + ".jpeg")
        image = image / 255.0
        image = image - pixel_means
        label = labels_dict[image_name.strip(".jpeg")]
        images.append(image)
        labels.append(label)
    labels = np.asarray(labels)
    images = np.asarray(images)
    n = settings.MINI_BATCH_SIZE * settings.TRAIN_PERC
    train_x = images[0:n]
    train_y = labels[0:n]
    test_x = images[n:]
    test_y = labels[n:]
    train_set = [train_x, train_y]
    test_set = [test_x, test_y]
    return [train_set, test_set]

#This will return an estimate of the mean value and spread
def get_pixel_means(image_names, input_dir, h, w, d=1):
    limit = 1000
    running_means = []
    image_matrix = np.zeros((limit, h*w, d))
    for i, image_name in enumerate(image_names):
        image = scipy.misc.imread(input_dir + "/" + image_name)
        image = image / 255.0
        image = image.reshape(h*w, d)
        image_matrix[i%limit, :, :] = image
        if (i+1) % limit == 0:
            current_mean = image_matrix.mean(axis=0)
            running_means.append([current_mean, limit])
            image_matrix = np.zeros((limit, h*w, d))
    x = len(image_names) % limit
    if x > 0:
        image_matrix = image_matrix[0:x, :, :]
        current_mean = image_matrix.mean(axis=0)
        running_means.append([current_mean, x])
    n = len(image_names)
    mean = reduce(lambda x,y: x + y[1]*y[0], [0]+running_means) / n
    return mean

#Iterates through each image passed in and turns it grey
def turn_images_grey(image_names, input_dir, output_dir):
    for image_name in image_names:
        image = scipy.misc.imread(input_dir + "/" + image_name)
        image = rgb2gray(image)
        scipy.misc.imsave(output_dir + "/" + image_name, image)

#Ensure folders exist to output the minibatches and pixel averages
if not os.path.isdir(settings.PROCESSED_TRAIN_DIR):
    os.makedirs(settings.PROCESSED_TRAIN_DIR)
if not os.path.isdir(settings.PARAMS_DIR):
    os.makedirs(settings.PARAMS_DIR)
if not os.path.isdir(settings.GREY_TRAIN_DIR):
    os.makedirs(settings.GREY_TRAIN_DIR)

#Let's get all the image names along with a map to their labels
image_names = [i for i in os.listdir(settings.RESIZED_TRAIN_DIR) if ".jpeg" in i]
labels_dict = load_labels(settings.TRAIN_LABELS)

#There's a class imbalance in the dataset so we want to group images by class
label_count_dict = Counter(labels_dict.values())
class_buckets = {}
for i in xrange(5):
    class_buckets[i] = [img for img in labels_dict if labels_dict[img]==i]

#Grayscale the images if that option is turned on
if settings.GREY:
    print "Greyscaling images..."
    clear_folder(settings.GREY_TRAIN_DIR)
    turn_images_grey(image_names, settings.RESIZED_TRAIN_DIR, settings.GREY_TRAIN_DIR)

#Let's get the average values for each pixel in the training set
pixel_means = 0
if settings.CENTER_AND_SCALE:
    print "Calculating pixel averages..."
    h, w, d = settings.IMAGE_SIZE
    if settings.GREY:
        pixel_means = get_pixel_means(image_names, settings.GREY_TRAIN_DIR, h, w)
        pixel_means = pixel_means.reshape(h, w)
    else:
        pixel_means = get_pixel_means(image_names, settings.RESIZED_TRAIN_DIR, h, w, d)
        pixel_means = pixel_means.reshape(h, w, d)

#Create mini batches of images using an equal amount of images from each class
#This is done round robin style.  Then save them to disk
print "Creating pickled minibatches"
clear_folder(settings.PROCESSED_TRAIN_DIR)
n_per_class = settings.MINI_BATCH_SIZE / len(class_buckets.keys())
n_batches = min(label_count_dict.values()) / n_per_class
for i in xrange(n_batches):
    batch_images = []
    for j in xrange(n_per_class):
        batch_images += [class_buckets[k].pop() for k in xrange(5)]
    if settings.GREY:
        datasets = create_dataset(batch_images, labels_dict, pixel_means, settings.GREY_TRAIN_DIR)
    else:
        datasets = create_dataset(batch_images, labels_dict, pixel_means, settings.RESIZED_TRAIN_DIR)
    f = open(settings.PROCESSED_TRAIN_DIR + "/set_0%i"%i, "wb")
    cPickle.dump(datasets, f)
    f.close()

#Save our pixel means for when we'll need to apply it to test images
np.save(settings.PIXEL_MEANS_FILE, pixel_means)





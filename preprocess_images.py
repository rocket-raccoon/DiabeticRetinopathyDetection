######################################################
# This script takes all the images and does some     #
# basic preprocessing so that the deep learning goes #
# more smoothly later.  This includes:               #
# 1)  Mean subtraction                               #
# 2)  Standardizing variance                         #
# 3)  Grayscaling                                    #
# 4)  normalization of values between [0,1]          #
# 5)  Pickling data                                  #
# 6)  Making the minibatches have equal class values #
######################################################

#Dependencies
import os
import scipy.misc
import numpy as np
import settings
import csv
import cPickle

#Turns the image matrix into greyscale
def rgb2gray(image):
    return np.dot(image[:,:,:3], [0.299, 0.587, 0.144])

#Load the training labels as a dictionary of (image_name, label) pairs
def load_labels():
    with open(settings.TRAIN_LABELS, 'rb') as csvfile:
        label_reader = csv.reader(csvfile, delimiter=",")
        next(label_reader, None)
        labels_dict = {name: int(label) for (name, label) in label_reader}
    return labels_dict

#Create a training and test set from a small subset of indices
def create_dataset(image_names, labels_dict, col_means, col_spread):
    images = []
    labels = []
    for image_name in image_names:
        image = scipy.misc.imread(settings.RESIZED_TRAIN_DIR + "/" + image_name)
        image = image / 255.0
        if settings.CENTER_AND_SCALE:
           image = image - col_means
           #image = image / col_spread
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
def get_center(image_names, h, w, d):
    limit = 1000
    running_means = []
    running_stds = []
    image_matrix = np.zeros((limit, h*w, d))
    for i, image_name in enumerate(image_names):
        image = scipy.misc.imread(settings.RESIZED_TRAIN_DIR + "/" + image_name)
        image = image / 255.0
        image = image.reshape(h*w, d)
        image_matrix[i%limit,:,:] = image
        if (i+1) % limit == 0:
            current_std = image_matrix.std(axis=0)
            current_mean = image_matrix.mean(axis=0)
            running_means.append([current_mean, limit])
            running_stds.append([current_std, limit])
            image_matrix = np.zeros((limit, h*w, d))
    x = len(image_names) % limit
    if x > 0:
        image_matrix = image_matrix[0:x, :, :]
        current_std = image_matrix.std(axis=0)
        current_mean = image_matrix.mean(axis=0)
        running_means.append([current_mean, x])
        running_stds.append([current_std, x])
    n = len(image_names)
    mean = reduce(lambda x,y: x + y[1]*y[0], [0]+running_means) / n
    std = reduce(lambda x,y: x + y[1]*y[0], [0]+running_stds) / n
    return mean, std

#Check if the directory where the pickled sets will go to exists
if not os.path.isdir(settings.PROCESSED_TRAIN_DIR):
    os.makedirs(settings.PROCESSED_TRAIN_DIR)

#There's a class imbalance, so we want to count how many of each class exist
f = open(settings.TRAIN_LABELS, "rb")
reader = csv.reader(f, delimiter=",")
label_count_dict = {}
reader.next()
for image, label in reader:
    if label not in label_count_dict:
        label_count_dict[label] = 0
    else:
        label_count_dict[label] += 1

#Let's get all the image names along with their labels
image_names = [i for i in os.listdir(settings.RESIZED_TRAIN_DIR) if ".jpeg" in i]
labels_dict = load_labels()

#Now let's seperate the image names into buckets corresponding to class
class_buckets = {}
for image_name in image_names:
    label = labels_dict[image_name.strip(".jpeg")]
    if label not in class_buckets:
        class_buckets[label] = []
    class_buckets[label].append(image_name)

#Let's get the center and spread for each pixel
if settings.CENTER_AND_SCALE:
    h,w,d = settings.IMAGE_SIZE
    col_means, col_spread = get_center(image_names, h, w, d)
    col_means = col_means.reshape(h, w, d)
    col_spread = col_spread.reshape(h, w, d)

#Create mini batches of images using an equal amount of images from each class
#This is done round robin style
#Afterwards, pickle the dataset
n_per_class = settings.MINI_BATCH_SIZE / len(class_buckets.keys())
n_batches = min(label_count_dict.values()) / n_per_class
for i in xrange(n_batches):
    print i
    batch_images = []
    for j in xrange(n_per_class):
        batch_images.append(class_buckets[0].pop())
        batch_images.append(class_buckets[1].pop())
        batch_images.append(class_buckets[2].pop())
        batch_images.append(class_buckets[3].pop())
        batch_images.append(class_buckets[4].pop())
    dataset = create_dataset(batch_images, labels_dict, col_means, col_spread)
    f = open(settings.PROCESSED_TRAIN_DIR + "/set_0%i"%i, "wb")
    cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)
    f.close()





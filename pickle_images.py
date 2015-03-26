#Dependencies
import cPickle
import os
import random
import numpy as np
import scipy.misc
import settings
import csv

#Load the training labels as a dictionary of (image_name, label) pairs
def load_labels():
    with open(settings.TRAIN_LABELS, 'rb') as csvfile:
        label_reader = csv.reader(csvfile, delimiter=",")
        next(label_reader, None)
        labels_dict = {name: int(label) for (name, label) in label_reader}
    return labels_dict

#Create a training and test set from a small subset of indices
def create_dataset(image_names, labels_dict):
    images = []
    labels = []
    for image_name in image_names:
        image = scipy.misc.imread(settings.PROCESSED_TRAIN_DIR + "/" + image_name)
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

#Check if the directory where the pickled sets will go to exists
if not os.path.isdir(settings.PICKLED_SETS_DIR):
    os.makedirs(settings.PICKLED_SETS_DIR)

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
image_names = [i for i in os.listdir(settings.PROCESSED_TRAIN_DIR) if ".jpeg" in i]
labels_dict = load_labels()

#Now let's seperate the image names into buckets corresponding to class
class_buckets = {}
for image_name in image_names:
    label = labels_dict[image_name.strip(".jpeg")]
    if label not in class_buckets:
        class_buckets[label] = []
    class_buckets[label].append(image_name)

#Create mini batches of images using an equal amount of images from each class
#This is done round robin style
#Afterwards, pickle the dataset
n_per_class = settings.MINI_BATCH_SIZE / len(class_buckets.keys())
n_batches = min(label_count_dict.values()) / n_per_class
for i in xrange(n_batches):
    batch_images = []
    for j in xrange(n_per_class):
        batch_images.append(class_buckets[0].pop())
        batch_images.append(class_buckets[1].pop())
        batch_images.append(class_buckets[2].pop())
        batch_images.append(class_buckets[3].pop())
        batch_images.append(class_buckets[4].pop())
    dataset = create_dataset(batch_images, labels_dict)
    f = open(settings.PICKLED_SETS_DIR + "/set_0%i"%i, "wb")
    cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


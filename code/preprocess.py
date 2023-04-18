import gzip
import pickle

import numpy as np

"""
TODO: 
Same as HW1. Feel free to copy and paste your old implementation here.
It's a good time to vectorize it, while you're at it!
No need to include CIFAR-specific methods.
"""

def get_data_MNIST(subset, data_path="../data"):
    """
    :param subset: string indicating whether we want the training or testing data 
        (only accepted values are 'train' and 'test')
    :param data_path: directory containing the training and testing inputs and labels
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    ## TODO: read the image file and normalize, flatten, and type-convert image
    image = np.zeros((num_examples,784),dtype=np.uint8)
    with open(inputs_file_path,'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)# it has 16 byte header
        digitBuffer = bytestream.read(784) #image consists 784 single bytes
        idx = 0
        while digitBuffer:
            digitArray = np.frombuffer(digitBuffer, dtype=np.uint8)
            image[idx,:] = digitArray
            idx = idx+1
            digitBuffer = bytestream.read(784)
    #Normalize
    image = image.astype(np.float32)/255.0 #change type and range from 0-1

    ## TODO: read the label file
    label = np.zeros((num_examples),dtype=np.uint8)
    with open(labels_file_path,'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)# it has 8 byte header
        labelBuffer = bytestream.read(1)
        idx = 0
        while labelBuffer:
            labelArray = np.frombuffer(labelBuffer, dtype=np.uint8)
            label[idx] = labelArray
            idx = idx+1
            labelBuffer = bytestream.read(1)

    return image, label
    #pass
    
## THE REST ARE OPTIONAL!

'''
def shuffle_data(image_full, label_full, seed):
    pass
    
def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    pass
'''

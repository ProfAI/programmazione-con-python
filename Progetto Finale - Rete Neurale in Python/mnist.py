import numpy as np
import os
import struct

def load_mnist(path="/"):

    train_labels_path = os.path.join(path,"train-labels.idx1-ubyte")
    train_images_path = os.path.join(path,"train-images.idx3-ubyte")

    test_labels_path = os.path.join(path,"t10k-labels.idx1-ubyte")
    test_images_path = os.path.join(path,"t10k-images.idx3-ubyte")

    labels_path = [train_labels_path, test_labels_path]
    images_path = [train_images_path, test_images_path]

    labels = []
    images = []

    for path in zip(labels_path, images_path):

        with open(path[0],'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            lb = np.fromfile(lbpath, dtype=np.uint8)
            labels.append(lb)

        with open(path[1], 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images.append(np.fromfile(imgpath, dtype=np.uint8).reshape(len(lb), 784))

    return images[0], images[1], labels[0], labels[1]

import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        self.txt_file = txt_file
        self.num_classes = num_classes
        self._read_txt_file()
        self.data_size = len(self.labels)

        if shuffle:
            self._shuffle_lists()

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
	
        if mode == 'training':
            data = data.map(self._parse_function_train)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        data = data.batch(batch_size)
        self.data = data

    def _read_txt_file(self):
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, MEAN)
        img_bgr = img_centered[:, :, ::-1]

        return filename, img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, MEAN)
        img_bgr = img_centered[:, :, ::-1]

        return filename, img_bgr, one_hot

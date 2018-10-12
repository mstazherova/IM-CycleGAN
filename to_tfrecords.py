import os
import io
import random
import cv2
import argparse

import tensorflow as tf


def reader(path, shuffle=True):
    files = []

    for img_file in os.scandir(path):
        if img_file.name.lower().endswith('.jpg', ) and img_file.is_file():
            files.append(img_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images. Make the randomization repeatable.
        shuffled_index = list(range(len(files)))
        random.shuffle(files)

        files = [files[i] for i in shuffled_index]

    return files


def writer(in_path, out_prefix):
    """Convert training and testing images to tfrecords files."""

    as_bytes = lambda data: tf.train.Feature(bytes_list=
                                             tf.train.BytesList(value=[data]))
    # Create an example protocol buffer & feature
    as_example = lambda data: tf.train.Example(
        features=tf.train.Features(feature=
        {'image/encoded_image': as_bytes((data))}))
    
    for sub in ['trainA', 'trainB', 'testA', 'testB']:
        indir = os.path.join(in_path, sub)
        outfile = os.path.abspath('{}_{}.tfrecords'.format(out_prefix, sub))
        files = reader(indir)

        record_writer = tf.python_io.TFRecordWriter(outfile)

        for i, img_path in enumerate(files):
            image = cv2.imread(img_path)
            encoded_image = cv2.imencode('.jpg', image)[1].tostring()
            example = as_example(encoded_image)
            record_writer.write(example.SerializeToString())

            if i % 100 == 0:
                print('{}: Processed {}/{}.'.format(sub, i, len(files)))
        print('Done.')
        record_writer.close()


if __name__ == "__main__":
    DATA_PATH = 'data/horse2zebra/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', default=DATA_PATH, 
                        help='Location of the data.')
    parser.add_argument('-o', '--out_prefix', default=DATA_PATH,
                        help='Prefix path for output tfrecords files')
    parser.add_argument('-s', '--seed', type=int, default=4321, help='Random seed to ensure repeatable shuffling')
    args = parser.parse_args()

    writer(args.in_path, args.out_prefix)

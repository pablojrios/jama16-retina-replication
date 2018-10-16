import tensorflow as tf
import os
from random import shuffle
import numpy as np

BRIGHTNESS_MAX_DELTA = 0.125
SATURATION_LOWER = 0.5
SATURATION_UPPER = 1.5
HUE_MAX_DELTA = 0.2
CONTRAST_LOWER = 0.5
CONTRAST_UPPER = 1.5

def _tfrecord_dataset_from_folder(folder, ext='.tfrecord'):
    tfrecords = [os.path.join(folder, n)
                 for n in os.listdir(folder) if n.endswith(ext)]
    return tf.data.TFRecordDataset(tfrecords)


def _decode_and_length_map(encoded_string):
    decoded = tf.decode_raw(encoded_string, out_type=tf.uint8)
    return decoded, tf.shape(decoded)[0]


def _parse_example(proto, num_channels, image_data_format, normalization_fn, data_augmentation=False):
    features = {"image/fileid": tf.FixedLenFeature((), tf.int64),
                "image/encoded": tf.FixedLenFeature((), tf.string),
                "image/format": tf.FixedLenFeature((), tf.string),
                "image/class/label": tf.FixedLenFeature((), tf.int64),
                "image/height": tf.FixedLenFeature((), tf.int64),
                "image/width": tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(proto, features)

    image = tf.image.decode_jpeg(parsed["image/encoded"], num_channels)

    if data_augmentation:
        # Apply data augmentations randomly.
        augmentations = [
            {'fn': tf.image.random_flip_left_right},
            {'fn': tf.image.random_brightness,
             'args': [BRIGHTNESS_MAX_DELTA]},
            {'fn': tf.image.random_saturation,
             'args': [SATURATION_LOWER, SATURATION_UPPER]},
            {'fn': tf.image.random_hue,
             'args': [HUE_MAX_DELTA]},
            {'fn': tf.image.random_contrast,
             'args': [CONTRAST_LOWER, CONTRAST_UPPER]}]

        # shuffle(augmentations)

        for aug in augmentations:
            if 'args' in aug:
                image = aug['fn'](image, *aug['args'])
            else:
                image = aug['fn'](image)

    # Standardize image.
    # image = tf.image.per_image_standardization(image)
    image = normalization_fn(image)

    if image_data_format == 'channels_first':
        image = tf.transpose(image, [2, 0, 1])

    label = tf.cast(
        tf.reshape(parsed["image/class/label"], [-1]),
        tf.float32)

    fileid = tf.cast(
        tf.reshape(parsed["image/fileid"], [-1]),
        tf.float32)

    return image, label, fileid


def initialize_dataset(image_dir, batch_size, num_epochs=1,
                       num_workers=1, prefetch_buffer_size=None,
                       shuffle_buffer_size=None,
                       normalization_fn=tf.image.per_image_standardization,
                       image_data_format='channels_last',
                       num_channels=3, data_augmentation=False):
    # Retrieve data set from pattern.
    dataset = _tfrecord_dataset_from_folder(image_dir)

    dataset = dataset.map(
        lambda e: _parse_example(
            e, num_channels, image_data_format, normalization_fn, data_augmentation),
        num_parallel_calls=num_workers)

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    if prefetch_buffer_size is not None:
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset

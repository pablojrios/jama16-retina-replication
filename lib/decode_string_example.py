import tensorflow as tf
import numpy as np

def _decode_and_length_map(encoded_string):
    decoded = tf.decode_raw(encoded_string, out_type=tf.uint8)
    return decoded, tf.shape(decoded)[0]


inputs = tf.constant(["aaa", "bbbbbbbb", "abcde", "123456"], dtype=tf.string)
dataset = (tf.data.Dataset.from_tensor_slices(inputs)
           .map(_decode_and_length_map)
           .padded_batch(batch_size=2, padded_shapes=([None], [])))
batch_op = dataset.make_one_shot_iterator().get_next()
with tf.Session() as session:
    a = session.run(batch_op)
    b = session.run(batch_op)
    print(a[0][0][:a[1][0]].tostring().decode('utf-8'))
    print(a[0][1][:a[1][1]].tostring().decode('utf-8'))
    print(b[0][0][:b[1][0]].tostring().decode('utf-8'))
    print(b[0][1][:b[1][1]].tostring().decode('utf-8'))
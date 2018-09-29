"""Usage example"""

# These can be any tensors of matching type and dimensions.
images = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
labels = tf.placeholder(tf.uint64, shape=(None))

images, labels = augment(images, labels,
                         horizontal_flip=True, rotate=15, crop_probability=0.8, mixup=4)
# ... Now build your model and loss on top of images and labels
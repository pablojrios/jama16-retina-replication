import numpy as np
import tensorflow as tf
import os
import random
import sys
import argparse
import csv
from glob import glob
import lib.metrics
import lib.dataset
import lib.evaluation
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.activations import sigmoid


def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), name="Xent")


def display_model_params():
    total_params = np.sum([np.prod(v.shape) for v in tf.global_variables()])
    trainable_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    non_trainable_params = total_params - trainable_params

    print("Total params: {:,}".format(total_params.value))
    print("Trainable params: {:,}".format(trainable_params.value))
    print("Non-trainable params: {:,}".format(non_trainable_params.value))


print(f"Numpy version: {np.__version__}")
print(f"Tensorflow version: {tf.__version__}")

# hacer visiable la GPU 1050 Ti, TF v1.10 pide TF_MIN_GPU_MULTIPROCESSOR_COUNT >= 8
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "6"
# usar la GeForce GTX 1080 Ti
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# Con 0 usa GeForce GTX 1050 Ti
os.environ["CUDA_VISIBLE_DEVICES"]="1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(432)

# Various loading and saving constants.
default_train_dir = "./data/eyepacs/bin2/train"
default_val_dir = "./data/eyepacs/bin2/validation"
default_save_model_path = "./tmp/model"
default_save_summaries_dir = "./tmp/logs"
default_save_operating_thresholds_path = "./tmp/validation_op_pts.csv"
# optimizer values: 'nesterov_accelerated_gd', 'vanilla_sgd', 'adam'
default_optimizer = "nesterov_accelerated_gd"


parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "detection of diabetic retinopathy.")
parser.add_argument("-t", "--train_dir",
                    help="path to folder that contains training tfrecords",
                    default=default_train_dir)
parser.add_argument("-v", "--val_dir",
                    help="path to folder that contains validation tfrecords",
                    default=default_val_dir)
parser.add_argument("-sm", "--save_model_path",
                    help="path to where graph model should be saved",
                    default=default_save_model_path)
parser.add_argument("-ss", "--save_summaries_dir",
                    help="path to folder where summaries should be saved",
                    default=default_save_summaries_dir)
parser.add_argument("-so", "--save_operating_thresholds_path",
                    help="path to where operating points should be saved",
                    default=default_save_operating_thresholds_path)
# parser.add_argument("-sgd", "--vanilla_sgd", action="store_true",
#                     help="use vanilla stochastic gradient descent instead of "
#                          "nesterov accelerated gradient descent")
parser.add_argument("-opt", "--optimizer",
                    help="optimizer algorithms: nesterov_accelerated_gd (default), vanilla_sgd, rmsprop",
                    default=default_optimizer)
parser.add_argument("-ld", "--large_diameter", action="store_true",
                    help="diameter of fundus to 512 pixels")

args = parser.parse_args()
train_dir = str(args.train_dir)
val_dir = str(args.val_dir)
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)
optimizer_name = args.optimizer
large_diameter = bool(args.large_diameter)

print("""
Training images folder: {},
Validation images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Saving operating points at: {},
Optimizer: {},
Large diameter: {}
""".format(train_dir, val_dir, save_model_path, save_summaries_dir,
           save_operating_thresholds_path, optimizer_name, large_diameter))

# Various constants.
num_channels = 3
num_workers = 8

# Hyper-parameters for training.
learning_rate = 5e-3
momentum = 0.9  # Only used when not training with momentum optimizer
use_nesterov = True  # Only used when not training with momentum optimizer
if optimizer_name == 'rmsprop':
    train_batch_size = 64
else:
    train_batch_size = 32 if large_diameter else 64

# ver https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py
# como entrenan an Inception v3 network from scratch, usando RMSPropOptimizer

# Hyper-parameters for training (arXiv:1710.01711)
# – Input image resolution: 299 × 299
# – Learning rate: 0.001
# – Batch size: 32
# – Weight decay: 4 · 10−5
# Constants dictating the learning rate schedule.
rmsprop_learning_rate = 0.001
rmsprop_decay = 0.9                # Decay term for RMSProp.
rmsprop_momentum = 0.9             # Momentum in RMSProp.
rmsprop_epsilon = 1.0              # Epsilon term for RMSProp.
# Decay the learning rate exponentially based on the number of steps.
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

# – An Adam optimizer with β1 = 0.9, β2 = 0.999, and epsilon = 0.1
# adam_learning_rate = 1e-3
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 0.1

# Hyper-parameters for validation.
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
if optimizer_name == 'rmsprop':
    val_batch_size = 64
else:
    val_batch_size = 32 if large_diameter else 64
num_thresholds = 200
kepsilon = 1e-7

# no funcionó, se obtienen resultados diferentes en 2 corridas
# tf.set_random_seed(12345)

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) + [0.5]

# Buffer size for image shuffling.
shuffle_buffer_size = 1024 if large_diameter else 2048
prefetch_buffer_size = 2 * train_batch_size

# Set image datas format to channels first if GPU is available.
# Este post explica porqué channels_first is preferred !
# https://stackoverflow.com/questions/44774234/why-tensorflow-uses-channel-last-ordering-instead-of-row-major
if tf.test.is_gpu_available():
    print("Found GPU! Using channels first as default image data format.")
    image_data_format = 'channels_first'
    input_shape=(3,512,512) if large_diameter else (3,299,299)
else:
    input_shape=(512,512,3) if large_diameter else (299,299,3)
    image_data_format = 'channels_last'

# Set up a session and bind it to Keras.
sess = tf.Session()
tf.keras.backend.set_session(sess)
tf.keras.backend.set_learning_phase(True)
tf.keras.backend.set_image_data_format(image_data_format)

# Initialize each data set.
train_dataset = lib.dataset.initialize_dataset(
    train_dir, train_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels)

val_dataset = lib.dataset.initialize_dataset(
    val_dir, val_batch_size,
    num_workers=num_workers, prefetch_buffer_size=prefetch_buffer_size,
    shuffle_buffer_size=shuffle_buffer_size,
    image_data_format=image_data_format, num_channels=num_channels)

# Create initializable iterators.
iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes)

images, labels, fileids = iterator.get_next()
x = tf.placeholder_with_default(images, images.shape, name='x')
y = tf.placeholder_with_default(labels, labels.shape, name='y')
fileid = tf.placeholder_with_default(fileids, fileids.shape, name='fileid')

train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# Base model InceptionV3 without top and global average pooling.
base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights='imagenet', pooling='avg', input_tensor=x)

# hay que ajustar los pesos de todas las capas para obtener mejor AUC
# base_model.trainable = True
# set_trainable = False
# for layer in base_model.layers:
#     if layer.name == 'mixed9':
#         set_trainable = True
#         # pass
#     if set_trainable:
#         layer.trainable = True
#         print("setting layer {} trainable".format(layer.name))
#     else:
#         layer.trainable = False

# Define optimizer.
global_step = tf.Variable(0, dtype=tf.int32)

if optimizer_name == 'vanilla_sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# elif optimizer_name == 'adam':
#     optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate,
#                                        beta1=beta1, beta2=beta2, epsilon=epsilon)
elif optimizer_name == 'rmsprop':
    # Create an optimizer that performs gradient descent.
    optimizer = tf.train.RMSPropOptimizer(learning_rate=rmsprop_learning_rate,
                                    decay=rmsprop_decay,
                                    momentum=rmsprop_momentum,
                                    epsilon=rmsprop_momentum)
else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum,
        use_nesterov=use_nesterov)

# tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# from https://www.kaggle.com/shuyuan00/resnet50-with-0-985lb/code
#
# global_average_pooling2d (Globa (None, 2048)         0           mixed10[0][0]
# __________________________________________________________________________________________________
# fc1_Dense (Dense)               (None, 1024)         2098176     global_average_pooling2d[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 1024)         0           fc1_Dense[0][0]
# __________________________________________________________________________________________________
# fc2_Dense (Dense)               (None, 256)          262400      dropout_1[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 256)          0           fc2_Dense[0][0]
# __________________________________________________________________________________________________
# output_layer (Dense)            (None, 1)            257         dropout_2[0][0]
#
# x_model = base_model.output
# x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
# x_model = Dropout(0.5, name='dropout_1')(x_model)
# x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
# x_model = Dropout(0.5, name='dropout_2')(x_model)
# # Add dense layer with the same amount of neurons as labels.
# logits = Dense(units=1, name='output_layer')(x_model)
# # Get the predictions with a sigmoid activation function.
# predictions = tf.sigmoid(logits, name='predictions')


# Add dense layer with the same amount of neurons as labels.
logits = tf.layers.dense(base_model.output, units=1)

# Get the predictions with a sigmoid activation function.
predictions = tf.sigmoid(logits, name='predictions')

# hay que ajustar los pesos de todas las capas para obtener mejor AUC
# retina_model = Model(inputs=base_model.inputs, outputs=logits)
# retina_model.compile(optimizer=optimizer, loss=sigmoid_cross_entropy_with_logits)
# retina_model.summary()

# Retrieve loss of network using cross entropy.
mean_xentropy = sigmoid_cross_entropy_with_logits(y_true=y, y_pred=logits)


# train_op = optimizer.minimize(loss=mean_xentropy, global_step=global_step, var_list=retina_model.trainable_variables)
train_op = optimizer.minimize(loss=mean_xentropy, global_step=global_step)

# Print all of the operations in the default graph.
# g = tf.get_default_graph()
# print(g.get_operations())
# otra forma de imprimir todos los nodos (operaciones) del grafo
# [n.name for n in tf.get_default_graph().as_graph_def().node if "dense" in n.name]
# obtener un tensor del grafo por nombre
# tf.get_default_graph().get_tensor_by_name("dense/kernel:0")

# Metrics for finding best validation set.
tp, update_tp, reset_tp = lib.metrics.create_reset_metric(
    tf.metrics.true_positives_at_thresholds, scope='tp',
    labels=y, predictions=predictions, thresholds=thresholds)

fp, update_fp, reset_fp = lib.metrics.create_reset_metric(
    tf.metrics.false_positives_at_thresholds, scope='fp',
    labels=y, predictions=predictions, thresholds=thresholds)

fn, update_fn, reset_fn = lib.metrics.create_reset_metric(
    tf.metrics.false_negatives_at_thresholds, scope='fn',
    labels=y, predictions=predictions, thresholds=thresholds)

tn, update_tn, reset_tn = lib.metrics.create_reset_metric(
    tf.metrics.true_negatives_at_thresholds, scope='tn',
    labels=y, predictions=predictions, thresholds=thresholds)

confusion_matrix = lib.metrics.confusion_matrix(
    tp[-1], fp[-1], fn[-1], tn[-1], scope='confusion_matrix')

brier, update_brier, reset_brier = lib.metrics.create_reset_metric(
    tf.metrics.mean_squared_error, scope='brier',
    labels=y, predictions=predictions)
tf.summary.scalar('mse', brier)

auc, update_auc, reset_auc = lib.metrics.create_reset_metric(
    tf.metrics.auc, scope='auc', labels=y, predictions=predictions)
tf.summary.scalar('auc', auc)

specificities = tf.div(tn, tn + fp + kepsilon)
sensitivities = tf.div(tp, tp + fn + kepsilon)

# Merge all the summaries and write them out.
summaries_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(save_summaries_dir + "/train", graph=sess.graph)
# todo: test_writer is not used
# test_writer = tf.summary.FileWriter(save_summaries_dir + "/test")

def print_training_status(epoch, num_epochs, batch_num, xent, i_step=None):
    def length(x): return len(str(x))

    m = []
    m.append(
        f"Epoch: {{0:>{length(num_epochs)}}}/{{1:>{length(num_epochs)}}}"
        .format(epoch, num_epochs))
    m.append(f"Batch: {batch_num:>4}, Xent: {xent:6.4}")

    if i_step is not None:
        m.append(f"Step: {i_step:>10}")

    print(", ".join(m), end="\r")


# Add ops for saving and restoring all variables.
saver = tf.train.Saver()

# Initialize variables.
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Train for the specified amount of epochs.
# Can be stopped early if peak of validation auc (Area under curve)
#  is reached.
latest_peak_auc = 0
waited_epochs = 0
latest_absolute_peak_auc = 0 # siempre me quedo con el mejor modelo aunque la mejora sea < min_delta_auc

for epoch in range(num_epochs):
    print("")

    # Start training.
    tf.keras.backend.set_learning_phase(True)
    sess.run(train_init_op)
    batch_num = 0

    # Track brier score for an indication on convergance.
    sess.run(reset_brier)

    try:
        while True:
            # Optimize cross entropy.
            i_global, batch_xent, *_ = sess.run(
                [global_step, mean_xentropy, train_op, update_brier])

            # Print a nice training status.
            print_training_status(
                epoch, num_epochs, batch_num, batch_xent, i_global)

            # Report summaries.
            batch_num += 1

    except tf.errors.OutOfRangeError:
        # Retrieve training brier score.
        train_brier = sess.run(brier)
        print("\nEnd of epoch {0}! (Brier: {1:8.6})".format(epoch, train_brier))

    # Perform validation.
    val_auc = lib.evaluation.perform_test(
        sess=sess, init_op=val_init_op,
        summary_writer=train_writer, epoch=epoch)

    # check for early stop
    if val_auc > latest_absolute_peak_auc:
        latest_absolute_peak_auc = val_auc
        print(f"New peak auc reached: {val_auc:10.8}")

        # Save the model weights.
        saver.save(sess, save_model_path)

    if val_auc < latest_peak_auc + min_delta_auc:
        # Stop early if peak of val auc has been reached.
        # If it is lower than the previous auc value, wait up to `wait_epochs`
        #  to see if it does not increase again.

        if wait_epochs == waited_epochs:
            print("Stopped early at epoch {0} with saved peak auc {1:10.8}"
                  .format(epoch+1, latest_absolute_peak_auc))
            break

        waited_epochs += 1
    else:
        latest_peak_auc = val_auc

        # Reset waited epochs.
        waited_epochs = 0

# Load the saved best meta graph and restore variables from that checkpoint.
saver = tf.train.import_meta_graph("{}.meta".format(save_model_path))
saver.restore(sess, save_model_path)

# Get predictions of all data of our training set.
tf.keras.backend.set_learning_phase(False)
sess.run([train_init_op, reset_tp, reset_fp, reset_fn, reset_tn])

try:
    while True:
        # Update all confusion metrics for each batch.
        sess.run([update_tp, update_fp, update_fn, update_tn])

except tf.errors.OutOfRangeError:
    pass

# Write sensitivities and specificities to file.
with open(save_operating_thresholds_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['threshold', 'specificity', 'sensitivity'])

    train_specificities, train_sensitivities = \
        sess.run([specificities, sensitivities])

    for idx in range(num_thresholds):
        writer.writerow([
            "{:0.4f}".format(x) for x in [
                thresholds[idx], train_specificities[idx],
                train_sensitivities[idx]]])

# Close the session.
sess.close()
sys.exit(0)

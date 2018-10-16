import tensorflow as tf


def generate_thresholds(num_thresholds, kepsilon=1e-7):
    thresholds = [
        (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
    ]
    return [0.0 - kepsilon] + thresholds + [1.0 - kepsilon]

# http://ronny.rest/blog/post_2017_09_11_tf_metrics/
def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope):
        metric_op, update_op = metric(**metric_args)
        # vars = tf.contrib.framework.get_variables(
        #    s, collection=tf.GraphKeys.LOCAL_VARIABLES)
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        # las variables son tp/reset, fp/reset, fn/reset, tn/reset, brier/reset, auc/reset
        reset_op = tf.variables_initializer(running_vars, name='reset')
    return metric_op, update_op, reset_op


def confusion_matrix(tp, fp, fn, tn, num_labels=1, scope='confusion_matrix'):
    with tf.variable_scope(scope) as s:
        return tf.cast(tf.reshape(tf.stack([tp, fp, fn, tn], 0),
                                  [num_labels, 2, 2],
                                  name='confusion_matrix'), dtype=tf.int32)

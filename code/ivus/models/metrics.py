import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

def f1(y_true, y_pred):
    _y_pred = tf.cast(y_pred + 0.5, tf.int32)
    _y_true = tf.cast(y_true + 0.5, tf.int32)

    tp = K.sum(K.cast(_y_true*_y_pred, 'float'), axis=1)
    # tn = K.sum(K.cast((1-_y_true)*(1-_y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-_y_true)*_y_pred, 'float'), axis=1)
    fn = K.sum(K.cast(_y_true*(1-_y_pred), 'float'), axis=1)

    p = tf.where(tf.equal(tp+fp, 0), tf.ones_like(tp), tp/(tp+fp))
    r = tf.where(tf.equal(tp+fn, 0), tf.ones_like(tp), tp/(tp+fn))
    f1 = tf.where(tf.equal(tp+fp+fn, 0), tf.ones_like(tp), 2*p*r/(p+r))
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def f1_loss(y_true, y_pred):
    f_measure = f1(y_true, y_pred)
    loss = binary_crossentropy(y_true, y_pred)
    return f_measure - 0.1*loss**2

import tensorflow as tf

def rmse_loss(rating,rate,length):
    error = tf.subtract(rating,rate)
    error = tf.square(error)
    error = tf.reduce_sum(error)
    error = tf.divide(error,length)
    error = tf.sqrt(error)
    return error

def auc(rating,rate):
    pass

def hr(rate, negative, length,k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    isIn = tf.cast(tf.equal(topk, 99), dtype=tf.float32)
    row = tf.reduce_sum(isIn,axis=1)
    all = tf.reduce_sum(row)
    return all/length
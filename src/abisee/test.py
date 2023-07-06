import tensorflow as tf

batch_size = 4
attn_len = 3
vsize = 6

batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)

enc_batch_extend_vocab = tf.Variable([[1, 3, 5], [2, 5, 1], [3, 1, 4], [2, 4, 6]])


indices = tf.stack((batch_nums, enc_batch_extend_vocab), axis=2)  # (batch_size, enc_t, 2)
print(batch_nums)
print(indices)
shape = [batch_size, vsize]
copy_dist = tf.Variable([[1, 0, 0], [0, 0.2, 0.8], [0.1, 0.3, 0.6], [0, 0.4, 0.6]])
attn_dists_projected = tf.scatter_nd(indices, copy_dist, shape)
print(attn_dists_projected)

# [4, 3, 2] -> [4, 3] -> [4, 6]

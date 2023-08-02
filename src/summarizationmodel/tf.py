import tensorflow as tf

embed = tf.keras.layers.Embedding(10, 3, mask_zero=True)
lstm = tf.keras.layers.LSTM(3, return_sequences=True, return_state=True)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = embed
        self.lstm = lstm

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embed(x)
        output, fwd, bwd = self.lstm(x)
        return output


data = tf.constant([[1, 2, 3, 0, 0, 0], [1, 3, 4, 0, 0, 0]])

model = Model()
print(model(data))

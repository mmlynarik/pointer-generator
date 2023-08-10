# import tensorflow as tf

# embed = tf.keras.layers.Embedding(10, 3, mask_zero=True)
# lstm = tf.keras.layers.LSTM(3, return_sequences=True, return_state=True)


# class Model(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.embed = embed
#         self.lstm = lstm

#     def __call__(self, x: tf.Tensor) -> tf.Tensor:
#         x = self.embed(x)
#         output, fwd, bwd = self.lstm(x)
#         return output


# data = tf.constant([[1, 2, 3, 0, 0, 0], [1, 3, 4, 0, 0, 0]])

# model = Model()
# print(model(data))

import torch
from torch import nn
from summarizationmodel.utils import gather_nd

torch.manual_seed(100)

lstm = nn.LSTM(input_size=5, hidden_size=5, num_layers=1)
lstm_cell = nn.LSTMCell(5, 5)

lstm_cell.weight_hh = lstm.weight_hh_l0
lstm_cell.bias_hh = lstm.bias_hh_l0
lstm_cell.weight_ih = lstm.weight_ih_l0
lstm_cell.bias_ih = lstm.bias_ih_l0

h0 = torch.Tensor([0.1, 0.2, 0.12, -0.3, 0.1]).unsqueeze(0)
c0 = torch.Tensor([0.1, 0.2, 0.12, -0.3, 0.1]).unsqueeze(0)
x = torch.randn(8, 5)

with torch.no_grad():
    h, c = h0, c0
    for i in range(8):
        out, (h, c) = lstm(x[i].unsqueeze(0), (h, c))
        print(f"{out=} {h=} {c=}")

    print("\n")
    h, c = h0, c0
    for i in range(8):
        h, c = lstm_cell(x[i].unsqueeze(0), (h, c))
        print(f"{h=} {c=}")


indices = torch.tensor(
    [
        [0, 2],
        [1, 3],
    ],
)
params = torch.tensor([[3, 4, 7, 5], [9, 1, 3, 6]])

print(gather_nd(params, indices))


indices = torch.tensor([[2], [3]])
print(torch.gather(params, 1, indices).squeeze())

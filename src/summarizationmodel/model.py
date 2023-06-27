import torch
from torch import nn
from lightning.pytorch import LightningModule


class PointerGeneratorModel(nn.Module):
    def __init__(
        # fmt: off
        self,
        hidden_dim: int = 256,                  # dimension of LSTM hidden states
        embedding_dim: int = 128,               # dimension of word embedding
        batch_size: int = 16,                   # minibatch size
        max_enc_steps: int = 400,               # max timesteps of encoder (max source text tokens)
        max_dec_steps: int = 100,               # max timesteps of decoder (max summary tokens)
        beam_size: int = 4,                     # beam size for beam search decoding.
        min_dec_steps: int = 35,                # Minimum seq length of generated summary. For decoding mode
        vocab_size: int = 50000,                # Maximum size of vocabulary
        rand_unif_init_mag: float = 0.02,       # magnitude for LSTM cells random uniform inititalization
        trunc_norm_init_std: float = 1e-4,      # std of trunc norm initialization, used for everything else
        max_grad_norm: float = 2.0,             # gradient clipping norm
        # fmt: on
    ):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps
        self.beam_size = beam_size
        self.min_dec_steps = min_dec_steps
        self.vocab_size = vocab_size
        self.rand_unif_init_mag = rand_unif_init_mag
        self.trunc_norm_init_std = trunc_norm_init_std
        self.max_grad_norm = max_grad_norm

    def encoder(self, encoder_inputs: torch.Tensor, seq_len: torch.Tensor):
        """Single-layer bi-directional LSTM encoder.

        Args:
        encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, embedding_dim].
        seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
        encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
        fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        cell_bw = nn.LSTMCell(
            self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True
        )
        # encoder_outputs, (fw_st, bw_st) = nn.bidirectional_dynamic_rnn(
        #     cell_fw, cell_bw, encoder_inputs, dtype=torch.float32, sequence_length=seq_len, swap_memory=True
        # )
        # encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        # concatenate the forwards and backwards states
        return  # encoder_outputs, fw_st, bw_st


class AbstractiveSummarizationModel(LightningModule):
    """
    A class to represent a seq-to-seq model for text summarization with pointer-generator mode and coverage.
    It's inspired by the paper https://arxiv.org/abs/1704.04368 and implementation by Abigail See, rewritten from TF1.0 into PyTorch and HuggingFace-based objects.
    """

    def __init__(self, model: PointerGeneratorModel, lr: float = 0.15):
        self.lr = lr                            # learning rate

    # inputs
    # self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name="enc_batch")
    # self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name="enc_lens")
    # self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name="enc_padding_mask")
    # self._enc_batch_extend_vocab = tf.placeholder(# tf.int32, [hps.batch_size, None])

    # self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name="dec_batch")
    # self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps])
    # self._max_art_oovs = tf.placeholder(tf.int32, [], name="max_art_oovs")
    # self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps])

    # self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name="prev_coverage") (decode)
# fmt: on

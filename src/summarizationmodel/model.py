from typing import Any
from dataclasses import dataclass

import torch
from torch import nn
from lightning.pytorch import LightningModule
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from summarizationmodel.datamodule.datamodule import pack_sequences, SummarizationDataModule
from summarizationmodel.datamodule.tokenizer import SummarizationTokenizerFast
from summarizationmodel.config import (
    END_TOKEN,
    MAX_DECODER_STEPS,
    MAX_ENCODER_STEPS,
    NON_PADDABLE_FEATURES,
    PADDABLE_FEATURES,
    START_TOKEN,
    TOKENIZER_DIR,
)


@dataclass
class LSTMStateTuple:
    """
    LSTM encoder final state with both attributes of shape [batch_size, actual_hidden_dim].
    Actual_hidden_dim = 2 * hidden_dim (for unreduced state) or hidden_dim (for reduced state).
    """
    hidden_state: torch.Tensor
    cell_state: torch.Tensor


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int, tokenizer: SummarizationTokenizerFast):
        super().__init__()
        vocab_size = tokenizer.backend_tokenizer.get_vocab_size()
        pad_token_id = tokenizer.pad_token_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_token_id)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs)


class PointerGeneratorEncoder(nn.Module):
    """Single-layer bi-directional LSTM encoder.

    Inputs: tensor of shape [batch_size, max_enc_steps, embedding_dim].
    Outputs:
        encoder_outputs of shape [batch_size, max_enc_steps, 2 * hidden_dim]
        unreduced lstm_state with attributes of shape [batch_size, 2 * hidden_dim]
    """
    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, LSTMStateTuple]:
        encoder_outputs, (hidden_state, cell_state) = self.encoder(inputs)
        hidden_state = torch.cat([direction for direction in hidden_state], axis=1).squeeze()
        cell_state = torch.cat([direction for direction in cell_state], axis=1).squeeze()
        lstm_state = LSTMStateTuple(hidden_state, cell_state)
        return encoder_outputs, lstm_state


class PointerGeneratorStateReducer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear_cell = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear_hidden = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, lstm_state: LSTMStateTuple) -> LSTMStateTuple:
        reduced_hidden_state = self.linear_hidden(lstm_state.hidden_state)
        reduced_cell_state = self.linear_cell(lstm_state.cell_state)
        return LSTMStateTuple(reduced_hidden_state, reduced_cell_state)


class PointerGeneratorSummarizatonModel(nn.Module):
    def __init__(
        # fmt: off
        self,
        hidden_dim: int,                        # dimension of LSTM hidden states
        embedding_dim: int,                     # dimension of word embedding
        batch_size: int,                        # minibatch size
        max_enc_steps: int,                     # max timesteps of encoder (max source text tokens)
        max_dec_steps: int,                     # max timesteps of decoder (max summary tokens)
        beam_size: int,                         # beam size for beam search decoding.
        min_dec_steps: int,                     # Minimum seq length of generated summary. For decoding mode
        tokenizer: SummarizationTokenizerFast,  # tokenizer used to get vocab
        rand_unif_init_mag: float = 0.02,       # magnitude for LSTM cells random uniform inititalization
        trunc_norm_init_std: float = 1e-4,      # std of trunc norm initialization, used for everything else
        max_grad_norm: float = 2.0,
        # gradient clipping norm
        # fmt: on
    ):
        super().__init__()
        self.embedding = Embedding(embedding_dim, tokenizer)
        self.encoder = PointerGeneratorEncoder(hidden_dim, embedding_dim)
        self.state_reducer = PointerGeneratorStateReducer(hidden_dim)

    def forward(self, inputs: dict[str, Any]):
        encoder_embeddings = self.embedding(inputs["encoder_input_ids"])
        encoder_output, lstm_state = self.encoder(encoder_embeddings)
        self.decoder_init_state = self.state_reducer(lstm_state)
        self.encoder_states = encoder_output
        return encoder_output


class AbstractiveSummarizationModel(LightningModule):
    """
    A class to represent a seq-to-seq model for text summarization with pointer-generator mode and coverage.
    It's inspired by the paper https://arxiv.org/abs/1704.04368 and implementation by Abigail See, rewritten from TF1.0 into PyTorch and HuggingFace-based objects.
    """

    def __init__(self, model: PointerGeneratorSummarizatonModel, lr: float = 0.15):
        self.lr = lr  # learning rate

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


def main():
    dm = SummarizationDataModule()
    dm.prepare_data()
    dm.setup()
    tokenizer = SummarizationTokenizerFast.from_pretrained(TOKENIZER_DIR)
    model = PointerGeneratorSummarizatonModel(256, 128, 16, 400, 100, 4, 35, tokenizer)

    for step, batch in enumerate(dm.train_dataloader()):
        if step == 0:
            output = model(batch)
            print(output.size())
            break


if __name__ == "__main__":
    main()

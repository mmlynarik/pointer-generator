from typing import Any, Optional, NamedTuple
from dataclasses import dataclass

import torch as pt
from torch import nn
from lightning.pytorch import LightningModule

from summarizationmodel.datamodule.datamodule import SummarizationDataModule
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


class LSTMState(NamedTuple):
    """Single-layer LSTM state consisting of two attributes of shape [batch_size, current_hidden_dim]."""

    hidden_state: pt.Tensor
    cell_state: pt.Tensor

    @property
    def concatenated(self) -> pt.Tensor:
        return pt.cat([self.hidden_state, self.cell_state], dim=1)


class SharedEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, pad_token_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_token_id)

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        return self.embedding(inputs)

    @property
    def embedding_dim(self) -> int:
        return self.embedding.embedding_dim


class PointerGeneratorStateReducer(nn.Module):
    """Dense layer reducing encoder final state dimension from 2 * hidden_dim down to hidden_dim."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, lstm_state: LSTMState) -> LSTMState:
        reduced_hidden_state = self.relu(self.hidden(lstm_state.hidden_state))
        reduced_cell_state = self.relu(self.cell(lstm_state.cell_state))
        return LSTMState(reduced_hidden_state, reduced_cell_state)


class PointerGeneratorEncoder(nn.Module):
    """Single-layer bi-directional LSTM encoder with PackedSequence inputs to LSTM instead of dynamic padding

    Inputs:
        input_ids: Article token ids of shape [batch_size, max_enc_steps]
        padding_mask: Boolean mask of shape [batch_size, max_enc_steps]
    Outputs:
        outputs: Tensor of shape [batch_size, max_enc_steps, 2 * hidden_dim]
        reduced_last_state: Last LSTMState with attributes of shape [batch_size, hidden_dim]
    """

    def __init__(self, hidden_dim: int, embedding: SharedEmbedding):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.state_reducer = PointerGeneratorStateReducer(hidden_dim)

    def forward(self, input_ids: pt.Tensor, padding_mask: pt.Tensor) -> tuple[pt.Tensor, LSTMState]:
        embeddings = self.embedding(input_ids)
        lengths = self.get_input_lengths(padding_mask)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, True, False)
        packed_outputs, (hidden_state, cell_state) = self.lstm(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        last_state = self.get_lstm_state_from_components(hidden_state, cell_state)
        reduced_last_state = self.state_reducer(last_state)
        return outputs, reduced_last_state

    def get_input_lengths(self, padding_mask: pt.Tensor) -> pt.Tensor:
        return padding_mask.sum(dim=1)

    def get_lstm_state_from_components(self, hidden_state: pt.Tensor, cell_state: pt.Tensor) -> LSTMState:
        hidden_state = pt.cat([direction for direction in hidden_state], axis=1).squeeze()
        cell_state = pt.cat([direction for direction in cell_state], axis=1).squeeze()
        return LSTMState(hidden_state, cell_state)


class BahdanauAttnWithCoverage(nn.Module):
    """Bahdanau attention layer used to compute context vector, attention distribution and coverage vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Wh = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.Ws = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.wc = nn.Linear(1, hidden_dim, bias=False)

    def forward(
        self,
        encoder_outputs: pt.Tensor,
        encoder_padding_mask: pt.Tensor,
        decoder_state: LSTMState,
        coverage: Optional[pt.Tensor] = None,
    ):
        coverage = pt.zeros_like(encoder_padding_mask) if coverage is None else coverage

        decoder_state = decoder_state.concatenated.unsqueeze(dim=1)
        encoder_padding_mask = encoder_padding_mask.unsqueeze(dim=2)
        coverage = coverage.unsqueeze(dim=2)

        encoder_features = self.Wh(encoder_outputs)
        decoder_features = self.Ws(decoder_state)
        coverage_features = self.wc(coverage)
        attn_scores = self.v(pt.tanh(encoder_features + decoder_features + coverage_features))  # B-L-1

        masked_scores = attn_scores.masked_fill(encoder_padding_mask, -float("inf"))
        attn_dist = nn.functional.softmax(masked_scores, dim=1)  # B-L-1

        context = attn_dist * encoder_outputs
        context = context.sum(dim=1)  # B-2H
        coverage += attn_dist

        return context, attn_dist.squeeze(), coverage


class BahdanauAttn(nn.Module):
    """Bahdanau attention layer used to compute context vector, attention distribution and coverage vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Wh = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.Ws = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: pt.Tensor,
        encoder_padding_mask: pt.Tensor,
        decoder_state: LSTMState,
        _: None,
    ):
        decoder_state = decoder_state.concatenated.unsqueeze(dim=1)
        encoder_padding_mask = encoder_padding_mask.unsqueeze(dim=2)

        encoder_features = self.Wh(encoder_outputs)
        decoder_features = self.Ws(decoder_state)
        attn_scores = self.v(pt.tanh(encoder_features + decoder_features))  # B-L-1

        masked_scores = attn_scores.masked_fill(encoder_padding_mask, -float("inf"))
        attn_dist = nn.functional.softmax(masked_scores, dim=1)  # B-L-1

        context = attn_dist * encoder_outputs
        context = context.sum(dim=1)  # B-2H

        return context, attn_dist.squeeze(), None


class VocabularyDistribution(nn.Module):
    """Reduce concatenated LSTM output and then apply final transformation to vocabulary distribution."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.reducer = nn.Linear(3 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        x = self.reducer(inputs)
        x = self.output(x)
        x = self.softmax(x)
        return x


class GenerationProbability(nn.Module):
    """Calculate generation probability from decoder state, input and and context vectors."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.wh = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.ws = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.wx = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: pt.Tensor, state: pt.Tensor, context: pt.Tensor) -> pt.Tensor:
        return self.sigmoid(self.wh(context) + self.ws(state) + self.wx(x))


class PointerGeneratorDecoder(nn.Module):
    """Single-layer unidirectional LSTM decoder with attention, pointer and coverage mechanism.

    Inputs:
        input_ids: Summary token ids of shape [batch_size, max_dec_steps]
        initial_state: Initial LSTMState with attributes of shape [batch_size, hidden_dim]
        encoder_outputs: Encoder outputs tensor of shape [batch_size, max_enc_seq_len, 2*hidden_dim]
        encoder_mask: Boolean padding mask of shape [batch_size, max_enc_seq_len]
        prev_coverage: Previous coverage vector, used during decoding of shape [batch_size, attention_length]
    Outputs:
        vocab_dists: Vocabulary distributions of shape [batch_size, max_dec_seq_len, vocab_size]. The words
        are in the order they appear in the tokenizer vocabulary.
        attn_dists: Attention distributions of shape [batch_size, max_dec_seq_len, max_enc_seq_len]
        pgens: Generation probabilities of shape [batch_size, max_dec_steps]
        coverage: Coverage vector of shape [batch_size, 2 * hidden_dim]
        state: Last decoder LSTMState
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        embedding: SharedEmbedding,
        initial_state_attention: bool = False,
        use_coverage: bool = False,
    ):
        super().__init__()
        self.embedding = embedding
        self.initial_state_attention = initial_state_attention

        self.reducer = nn.Linear(2 * hidden_dim + embedding.embedding_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.attention = BahdanauAttnWithCoverage(hidden_dim) if use_coverage else BahdanauAttn(hidden_dim)

        self.pgen = GenerationProbability(hidden_dim)
        self.vocab_dist = VocabularyDistribution(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: pt.Tensor,
        initial_state: LSTMState,
        encoder_outputs: pt.Tensor,
        encoder_mask: pt.Tensor,
        prev_coverage: Optional[pt.Tensor] = None,
    ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, LSTMState]:
        vocab_dists, attn_dists, pgens = [], [], []

        state: LSTMState = initial_state
        coverage = prev_coverage
        context = pt.zeros(encoder_outputs.shape[::2])

        if self.initial_state_attention:  # true in decode mode
            context, _, coverage = self.attention(encoder_outputs, encoder_mask, state, coverage)

        embeddings = self.embedding(input_ids)
        for step, step_embeddings in enumerate(embeddings.transpose(0, 1)):
            concatenated_input = pt.cat([step_embeddings, context], dim=1)
            x = self.reducer(concatenated_input)
            state = self.lstm_cell(x, state)
            state = LSTMState(state[0], state[1])

            if step == 0 and self.initial_state_attention:
                context, attn_dist, _ = self.attention(encoder_outputs, encoder_mask, state, coverage)
            else:
                context, attn_dist, coverage = self.attention(encoder_outputs, encoder_mask, state, coverage)

            concatenated_lstm_output = pt.cat([state.hidden_state, context], dim=1)
            attn_dists.append(attn_dist)
            pgens.append(self.pgen(x, state.concatenated, context))
            vocab_dists.append(self.vocab_dist(concatenated_lstm_output))

        vocab_dists = pt.stack(vocab_dists, dim=1)
        attn_dists = pt.stack(attn_dists, dim=1)
        pgens = pt.stack(pgens, dim=1)

        return vocab_dists, attn_dists, pgens, coverage, state


class PointerGeneratorSummarizatonModel(nn.Module):
    def __init__(
        # fmt: off
        self,
        hidden_dim: int,                        # dimension of LSTM hidden states
        embedding_dim: int,                     # dimension of word embedding
        beam_size: int,                         # beam size for beam search decoding.
        min_dec_steps: int,                     # Minimum seq length of generated summary. For decoding mode
        vocab_size: int,                        # vocab size
        pad_token_id: int,                      # pad token id
        rand_unif_init_mag: float = 0.02,       # magnitude for LSTM cells random uniform inititalization
        trunc_norm_init_std: float = 1e-4,      # std of trunc norm initialization, used for everything else
        max_grad_norm: float = 2.0,
        # gradient clipping norm
        # fmt: on
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = SharedEmbedding(embedding_dim, vocab_size, pad_token_id)
        self.encoder = PointerGeneratorEncoder(hidden_dim, self.embedding)
        self.decoder = PointerGeneratorDecoder(hidden_dim, vocab_size, self.embedding, False, False)

    def forward(self, inputs: dict[str, Any]) -> pt.Tensor:
        encoder_input_ids = inputs["encoder_input_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_input_ids = inputs["decoder_input_ids"]
        oovs = inputs["oovs"]
        encoder_inputs_extvoc = inputs["encoder_inputs_extvoc"]

        encoder_outputs, encoder_reduced_last_state = self.encoder(encoder_input_ids, encoder_padding_mask)
        vocab_dists, attn_dists, pgens, coverage, state = self.decoder(
            decoder_input_ids, encoder_reduced_last_state, encoder_outputs, encoder_padding_mask, None
        )
        max_article_oovs = self.get_max_article_oovs(oovs)
        return self.calc_final_dists(vocab_dists, attn_dists, pgens, max_article_oovs, encoder_inputs_extvoc)

    def get_max_article_oovs(self, oovs: list[list[int]]) -> int:
        return max(len(example_oovs) for example_oovs in oovs)

    def calc_final_dists(
        self,
        vocab_dists: pt.Tensor,
        attn_dists: pt.Tensor,
        pgens: pt.Tensor,
        max_article_oovs: int,
        encoder_inputs_extvoc: pt.Tensor,
    ) -> pt.Tensor:
        """Calculate final token distributions from vocab distributions and attention distributions by projecting the attn dists onto extended vocab size tensor and summing it with extended vocab dists.

        Args:
            vocab_dists: The vocabulary distributions of shape [batch_size, max_dec_steps, vocab_size]
            The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions of shape [batch_size, max_dec_seq_len, max_enc_seq_len].
            pgens: The generation probabilities of shape [batch_size, max_dec_seq_len].
            max_article_oovs: Maximum over each example in the batch number of in-article oovs.
            encoder_inputs_extvoc: Article extended-vocabulary token ids including OOV words of shape
            [batch_size, max_enc_seq_len].

        Returns:
        Final_dists: The final distributions of shape [batch_size, max_dec_steps, extended_vocab_size].
        """
        vocab_dists = pgens * vocab_dists
        attn_dists = (1 - pgens) * attn_dists

        batch_size, max_dec_steps = vocab_dists.shape[:2]
        extended_vocab_size = self.vocab_size + max_article_oovs
        extra_zeros = pt.zeros(batch_size, max_dec_steps, max_article_oovs)
        vocab_dists_extended = pt.concat([vocab_dists, extra_zeros], dim=2)

        index = pt.stack(max_dec_steps * [encoder_inputs_extvoc.long()], dim=1)
        projection = pt.zeros(batch_size, max_dec_steps, extended_vocab_size)
        attn_dists_projected = projection.scatter_add(2, index, attn_dists)
        return vocab_dists_extended + attn_dists_projected


class AbstractiveSummarizationModel(LightningModule):
    """
    A class to represent an abstractive seq2seq model for text summarization with pointer-generator network and coverage. It's inspired by the paper https://arxiv.org/abs/1704.04368 and implementation by Abigail See, rewritten from TF1.0 into PyTorch and Transformers libraries.
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

    model = PointerGeneratorSummarizatonModel(
        hidden_dim=256,
        embedding_dim=128,
        max_dec_steps=tokenizer.max_decoder_steps,
        vocab_size=tokenizer.backend_tokenizer.get_vocab_size(),
        beam_size=4,
        min_dec_steps=35,

        pad_token_id=tokenizer.pad_token_id,
    )

    for step, batch in enumerate(dm.train_dataloader()):
        if step == 0:
            final_dists = model(batch)
            print(final_dists.size())
            break


if __name__ == "__main__":
    main()

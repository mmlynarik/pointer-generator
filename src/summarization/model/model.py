from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import torch as pt
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn.functional import nll_loss

from summarization.datamodule.config import VOCAB_SIZE
from summarization.datamodule.datamodule import SummarizationDataModule
from summarization.utils import timeit


# Model and trainer config from research paper
@dataclass
class ModelConfig:
    hidden_dim: int = 256
    embedding_dim: int = 128
    beam_size: int = 4
    min_dec_steps: int = 35
    vocab_size: int = VOCAB_SIZE
    pad_token_id: int = 0
    use_coverage: bool = False
    learning_rate: float = 0.15
    adagrad_init_acc: float = 0.1
    cov_loss_weight: float = 1.0
    max_grad_norm: int = 2
    trunc_norm_init_std: float = 1e-4
    device: str = "cuda"


class LSTMState(NamedTuple):
    """Single-layer LSTM state consisting of two attributes of shape [batch_size, state_dim]."""

    hidden_state: pt.Tensor
    cell_state: pt.Tensor

    @property
    def concatenated(self) -> pt.Tensor:
        return pt.cat([self.hidden_state, self.cell_state], dim=1)

    @classmethod
    def from_tuple(cls, state: tuple[pt.Tensor, pt.Tensor]) -> "LSTMState":
        return cls(state[0], state[1])


class SharedEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, config.pad_token_id)

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

    def __init__(self, config: ModelConfig, embedding: SharedEmbedding):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.state_reducer = PointerGeneratorStateReducer(config.hidden_dim)

    def forward(self, input_ids: pt.Tensor, padding_mask: pt.Tensor) -> tuple[pt.Tensor, LSTMState]:
        embeddings = self.embedding(input_ids)
        lengths = self.get_input_lengths(padding_mask)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), True, False)
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

        masked_scores = attn_scores.masked_fill(encoder_padding_mask == 0, -float("inf"))
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
        masked_scores = attn_scores.masked_fill(encoder_padding_mask == 0, -float("inf"))
        attn_dist = nn.functional.softmax(masked_scores, dim=1)  # B-L-1

        context = attn_dist * encoder_outputs  # broadcasting multiplication
        context = context.sum(dim=1)  # B-2H

        return context, attn_dist.squeeze(), None


class VocabularyDistribution(nn.Module):
    """Reduce concatenated LSTM output and then apply output projection to vocabulary distribution."""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.reducer = nn.Linear(3 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs: pt.Tensor) -> pt.Tensor:
        x = self.reducer(inputs)
        x = self.output_projection(x)
        x = self.softmax(x)
        return x


class GenerationProbability(nn.Module):
    """Calculate generation probability from decoder state, decoder input and and context vector."""

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
        config: ModelConfig,
        embedding: SharedEmbedding,
        initial_state_attention: bool = False,
    ):
        super().__init__()
        self.embedding = embedding
        self.initial_state_attention = initial_state_attention
        self.device = config.device

        self.input_reducer = nn.Linear(2 * config.hidden_dim + embedding.embedding_dim, config.hidden_dim)
        self.lstm_cell = nn.LSTMCell(config.hidden_dim, config.hidden_dim)
        self.attention = (
            BahdanauAttnWithCoverage(config.hidden_dim)
            if config.use_coverage
            else BahdanauAttn(config.hidden_dim)
        )

        self.pgen = GenerationProbability(config.hidden_dim)
        self.vocab_dist = VocabularyDistribution(config.hidden_dim, config.vocab_size)

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
        context = pt.zeros(encoder_outputs.shape[::2], device=self.device)

        if self.initial_state_attention:  # true in decode mode
            context, _, coverage = self.attention(encoder_outputs, encoder_mask, state, coverage)

        embeddings = self.embedding(input_ids)
        for step, step_embeddings in enumerate(embeddings.transpose(0, 1)):
            concatenated_input = pt.cat([step_embeddings, context], dim=1)
            x = self.input_reducer(concatenated_input)
            state: tuple[pt.Tensor, pt.Tensor] = self.lstm_cell(x, state)
            state = LSTMState.from_tuple(state)

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


class PointerGeneratorSummarizationModel(nn.Module):
    """
    A class to represent an abstractive seq2seq text summarization model with pointer-generator network and coverage. It's inspired by the paper https://arxiv.org/abs/1704.04368 and implementation by Abigail See, rewritten from TF1.0 into PyTorch and Transformers libraries.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.use_coverage = config.use_coverage
        self.beam_size = config.beam_size
        self.min_dec_steps = config.min_dec_steps
        self.device = config.device
        self.trunc_norm_init_std = config.trunc_norm_init_std

        self.embedding = SharedEmbedding(config)
        self.encoder = PointerGeneratorEncoder(config, self.embedding)
        self.decoder = PointerGeneratorDecoder(config, self.embedding, False)
        self.init_weights()

    def forward(self, inputs: dict[str, Any]) -> tuple[pt.Tensor, pt.Tensor]:
        pt.cuda.empty_cache()

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
        final_dists = self.calc_final_dists(
            vocab_dists, attn_dists, pgens, max_article_oovs, encoder_inputs_extvoc
        )
        return final_dists, attn_dists

    def init_weights(self):
        """Tensorflow-like initialization of LSTM layer. Dense layers initialization inspired by paper."""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias_ih" in name:
                    nn.init.constant_(param.data, 0)
                    # Set forget-gate bias to 1
                    n = param.size(0)
                    param.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    nn.init.constant_(param.data, 0)
            elif "embedding" in name or "output_projection" in name or "state_reducer" in name:
                nn.init.trunc_normal_(param.data, std=self.trunc_norm_init_std)

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

        batch_size, max_dec_seq_len = vocab_dists.shape[:2]
        extended_vocab_size = self.vocab_size + max_article_oovs
        extra_zeros = pt.zeros(batch_size, max_dec_seq_len, max_article_oovs, device=self.device)
        vocab_dists_extended = pt.concat([vocab_dists, extra_zeros], dim=2)

        index = pt.stack(max_dec_seq_len * [encoder_inputs_extvoc.long()], dim=1)
        projection_base = pt.zeros(batch_size, max_dec_seq_len, extended_vocab_size, device=self.device)
        attn_dists_projected = projection_base.scatter_add(2, index, attn_dists)
        final_dists = vocab_dists_extended + attn_dists_projected
        return final_dists


class AbstractiveSummarizationModel(LightningModule):
    """Pytorch-Lightning encapsulation of Pointer-generator summarization model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.pad_token_id = config.pad_token_id
        self.model = PointerGeneratorSummarizationModel(config)
        self.adagrad_init_acc = config.adagrad_init_acc
        self.cov_loss_weight = config.cov_loss_weight
        self.max_grad_norm = config.max_grad_norm
        self.save_hyperparameters()
        self.metrics = {}

    def forward(self, batch: pt.Tensor) -> pt.Tensor:
        final_dists, _ = self.model(batch)
        return final_dists

    def _calc_primary_loss(self, final_dists: pt.Tensor, decoder_target_ids: pt.Tensor) -> pt.Tensor:
        """Calculate the model primary loss for batch from final token distributions and targets."""
        return pt.stack(
            [
                nll_loss(pt.log(example_probs), example_targets.long(), ignore_index=self.pad_token_id)
                for example_probs, example_targets in zip(final_dists, decoder_target_ids)
            ]
        ).mean()

    def _calc_coverage_loss(self, attn_dists: pt.Tensor, decoder_padding_mask: pt.Tensor) -> pt.Tensor:
        """
        Calculate the coverage loss from the attention distributions. First, calculate the coverage losses  for each decoder step and then doubly averaging the resulting tensor of shape [batch_size, dec_max_seq_len] using the decoder padding mask into a scalar.
        """
        coverage = pt.zeros(attn_dists.shape[::2])
        covlosses_per_dec_step = []
        for attn_dist_dec_step in attn_dists.transpose(0, 1):
            covloss = pt.minimum(attn_dist_dec_step, coverage).sum(dim=1)
            covlosses_per_dec_step.append(covloss)
            coverage += attn_dist_dec_step

        covlosses_per_dec_step = pt.stack(covlosses_per_dec_step, dim=1)
        covlosses_per_dec_step *= decoder_padding_mask
        covlosses_per_dec_step.sum(dim=1) / decoder_padding_mask.sum(dim=1)
        return covlosses_per_dec_step.mean()

    def training_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        decoder_padding_mask, decoder_target_ids = batch["decoder_padding_mask"], batch["decoder_target_ids"]
        final_dists, attn_dists = self.model(batch)
        bs = len(final_dists)  # batch_size to fix dataloader warning arising from str dtype in batch
        loss = self._calc_primary_loss(final_dists, decoder_target_ids)
        if self.model.use_coverage:
            coverage_loss = self._calc_coverage_loss(attn_dists, decoder_padding_mask)
            loss += self.cov_loss_weight * coverage_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        decoder_padding_mask, decoder_target_ids = batch["decoder_padding_mask"], batch["decoder_target_ids"]
        final_dists, attn_dists = self.model(batch)
        bs = len(final_dists)  # batch_size to fix dataloader warning arising from str dtype in batch
        loss = self._calc_primary_loss(final_dists, decoder_target_ids)
        if self.model.use_coverage:
            coverage_loss = self._calc_coverage_loss(attn_dists, decoder_padding_mask)
            loss += self.cov_loss_weight * coverage_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=bs)
        return loss

    def test_step(self, batch: dict[str, Any], _: int) -> STEP_OUTPUT:
        pass

    def configure_optimizers(self) -> Any:
        optimizer = pt.optim.Adagrad(
            self.parameters(), self.learning_rate, initial_accumulator_value=self.adagrad_init_acc
        )
        return optimizer

    def on_epoch_start(self):
        print(f"Global step: {self.trainer.global_step}\n")


@timeit
def main():
    datamodule = SummarizationDataModule(batch_size=16)
    datamodule.prepare_data()
    datamodule.setup()

    config = ModelConfig(device="cpu")
    module = AbstractiveSummarizationModel(config)
    for step, batch in enumerate(datamodule.train_dataloader()):
        final_dists, _ = module.model(batch)
        print(final_dists.shape)
        if step == 9:
            break


if __name__ == "__main__":
    main()

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from tokenizers import Tokenizer, processors
from transformers import BatchEncoding, PreTrainedTokenizerFast

from summarizationmodel.config import (
    END_TOKEN,
    MAX_DECODER_STEPS,
    MAX_ENCODER_STEPS,
    START_TOKEN,
    TOKENIZER_DIR,
)

TEXT = Union[str, Sequence[str]]
TokenizerFunction = Callable[[TEXT], BatchEncoding]
TruncationChecker = Callable[[TEXT], bool]
START_OR_END_TOKEN = Union[START_TOKEN, END_TOKEN]


PADDABLE_FEATURES = [
    "encoder_input_ids",
    "encoder_padding_mask",
    "decoder_input_ids",
    "decoder_padding_mask",
    "decoder_target_ids",
    "encoder_inputs_extvoc",
]

NON_PADDABLE_FEATURES = ["oovs"]


class SummarizationTokenizerFast(PreTrainedTokenizerFast):
    """Summarization task tokenizer capable of correctly tokenizing encoder and decoder inputs and targets."""

    def __init__(
        self,
        tokenizer_object: Tokenizer,
        model_input_names: list[str],
        model_max_length: int,
        pad_token: str,
        unk_token: str,
        bos_token: str,
        eos_token: str,
        max_encoder_steps: int = MAX_ENCODER_STEPS,
        max_decoder_steps: int = MAX_DECODER_STEPS,
        paddable_features: list[str] = PADDABLE_FEATURES,
        non_paddable_features: list[str] = NON_PADDABLE_FEATURES,
    ):
        super().__init__(
            tokenizer_object=tokenizer_object,
            model_max_length=model_max_length,
            model_input_names=model_input_names,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        self.max_encoder_steps = max_encoder_steps
        self.max_decoder_steps = max_decoder_steps
        self.paddable_features = paddable_features
        self.non_paddable_features = non_paddable_features

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str):
        pretrained_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        return cls(
            tokenizer_object=pretrained_tokenizer.backend_tokenizer,
            model_input_names=pretrained_tokenizer.model_input_names,
            model_max_length=pretrained_tokenizer.model_max_length,
            pad_token=pretrained_tokenizer.pad_token,
            unk_token=pretrained_tokenizer.unk_token,
            bos_token=pretrained_tokenizer.bos_token,
            eos_token=pretrained_tokenizer.eos_token,
        )

    def _apply_special_token_postprocessor(self, token: START_OR_END_TOKEN) -> "SummarizationTokenizerFast":
        """Apply a post-processor to backend tokenizer, adding either [START] or [END] special token."""
        special_token = (token, self.convert_tokens_to_ids(token))
        template = f"{token}:0 $A:0" if token == START_TOKEN else f"$A:0 {token}:0"
        post_processor = processors.TemplateProcessing(single=template, special_tokens=[special_token])
        self.backend_tokenizer.post_processor = post_processor
        return self

    def _apply_empty_postprocessor(self) -> "SummarizationTokenizerFast":
        """Remove a post-processor from backend tokenizer."""
        self.backend_tokenizer.post_processor = processors.TemplateProcessing()
        return self

    def _get_truncation_checker(self) -> TruncationChecker:
        """
        Truncation checker is used to decide how decoder targets will be encoded. Truncation will be applied if tokens (with added special start token) exceed batch max_decoder_steps.
        """
        checker = self._apply_special_token_postprocessor(START_TOKEN)

        def truncation_checker(single_text: str) -> bool:
            tokens = checker.tokenize(single_text, add_special_tokens=True)
            return len(tokens) > self.max_decoder_steps

        return truncation_checker

    def _get_batch_encoding_from_list(self, encodings: list[BatchEncoding]) -> BatchEncoding:
        return BatchEncoding(
            data={
                "input_ids": [item["input_ids"] for item in encodings],
                "attention_mask": [item["attention_mask"] for item in encodings],
                "oovs": [item["oovs"] for item in encodings] if "oovs" in encodings[0] else [],
            },
            encoding=encodings,
        )

    def generate_encoder_inputs(self, batch: TEXT) -> BatchEncoding:
        """
        Run __call__ method as an encoder tokenizer without adding any special tokens. Final step applies only truncation. Padding is deferred to collator function.
        """
        self._apply_empty_postprocessor()
        return self(batch, truncation=True, max_length=self.max_encoder_steps)

    def generate_decoder_inputs(self, batch: TEXT) -> BatchEncoding:
        """
        Run __call__ method as a decoder inputs tokenizer. Always adds START_TOKEN. Final step applies only truncation. Padding is deferred to collator function.
        """
        self._apply_special_token_postprocessor(START_TOKEN)
        return self(batch, truncation=True, max_length=self.max_decoder_steps)

    def generate_decoder_targets(self, batch: TEXT, oovs: list[list[str]]) -> BatchEncoding:
        """
        Run __call__ method as a decoder targets tokenizer. Adds END_TOKEN only if decoder input has not been truncated. Final step applies only truncation. Padding is deferred to collator function.
        """
        tokenizer_wo_special_token = deepcopy(self._apply_empty_postprocessor())
        tokenizer_with_end_token = deepcopy(self._apply_special_token_postprocessor(END_TOKEN))
        truncation_checker = deepcopy(self._get_truncation_checker())

        batch = [batch] if isinstance(batch, str) else batch
        encodings: list[BatchEncoding] = []
        for idx, text in enumerate(batch):
            words = self.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            words = words + [(END_TOKEN, ())] if len(words) < self.max_decoder_steps else words
            encoding = (
                tokenizer_wo_special_token(text, max_length=self.max_decoder_steps, truncation=True)
                if truncation_checker(text)
                else tokenizer_with_end_token(text)
            )
            input_ids = []
            for input_id, (word, _) in zip(encoding.input_ids, words):
                if input_id == self.unk_token_id:
                    if word in oovs[idx]:
                        input_ids.append(self.vocab_size + oovs[idx].index(word))
                    else:
                        input_ids.append(self.unk_token_id)
                else:
                    input_ids.append(input_id)

            encoding.data["input_ids"] = input_ids
            encodings.append(encoding)
        return self._get_batch_encoding_from_list(encodings)

    def generate_encoder_inputs_extend_vocab(self, batch: TEXT) -> BatchEncoding:
        """
        Run __call__ method as an extended-vocabulary encoder tokenizer considering OOV tokens per example without adding any special tokens. Final step applies only truncation. Padding is deferred to collator function.
        """
        self._apply_empty_postprocessor()
        batch = [batch] if isinstance(batch, str) else batch
        encodings: list[BatchEncoding] = []
        for text in batch:
            oovs, input_ids = [], []
            encoding = self(text, truncation=True, max_length=self.max_encoder_steps)
            words = self.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            for input_id, (word, _) in zip(encoding.input_ids, words):
                if input_id == self.unk_token_id:
                    if word not in oovs:
                        oovs.append(word)
                    input_ids.append(self.vocab_size + oovs.index(word))
                else:
                    input_ids.append(input_id)

            encoding.data["input_ids"], encoding.data["oovs"] = input_ids, oovs
            encodings.append(encoding)

        return self._get_batch_encoding_from_list(encodings)

    def prepare_model_inputs(self, batch: dict) -> dict:
        articles, abstracts = batch["article"], batch["highlights"]

        encoder_inputs = self.generate_encoder_inputs(articles)
        decoder_inputs = self.generate_decoder_inputs(abstracts)
        encoder_inputs_extend_vocab = self.generate_encoder_inputs_extend_vocab(articles)
        decoder_targets = self.generate_decoder_targets(abstracts, encoder_inputs_extend_vocab["oovs"])

        encoding = SummarizationBatchEncoding(
            encoder_inputs, decoder_inputs, decoder_targets, encoder_inputs_extend_vocab
        )
        return {
            **encoding.get_encoder_features(),
            **encoding.get_decoder_features(),
            **encoding.get_encoder_extvoc_features(),
        }


@dataclass
class SummarizationBatchEncoding:
    encoder_inputs: BatchEncoding
    decoder_inputs: BatchEncoding
    decoder_targets: BatchEncoding
    encoder_inputs_extend_vocab: BatchEncoding

    def get_encoder_features(self) -> dict:
        return {
            "encoder_input_ids": self.encoder_inputs["input_ids"],
            "encoder_padding_mask": self.encoder_inputs["attention_mask"],
        }

    def get_decoder_features(self) -> dict:
        return {
            "decoder_input_ids": self.decoder_inputs["input_ids"],
            "decoder_padding_mask": self.decoder_targets["attention_mask"],
            "decoder_target_ids": self.decoder_targets["input_ids"],
        }

    def get_encoder_extvoc_features(self) -> dict:
        return {
            "encoder_inputs_extvoc": self.encoder_inputs_extend_vocab["input_ids"],
            "oovs": self.encoder_inputs_extend_vocab["oovs"],
        }


def main():
    """Test tokenizer"""
    tokenizer = SummarizationTokenizerFast.from_pretrained(TOKENIZER_DIR)

    text = "Here is a short article."
    text_2 = "Here is a long article, which exceeds modddsdsdel madffdfx ldddength."
    batch = {"article": [text, text_2], "highlights": [text, text_2 + " bbbbbb"]}

    features = tokenizer.prepare_model_inputs(batch)
    print(f"Input texts:\n{batch} \n\nModel inputs:\n{features}")


if __name__ == "__main__":
    main()

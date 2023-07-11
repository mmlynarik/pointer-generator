from typing import Callable, Sequence, Union
from copy import deepcopy
from dataclasses import dataclass

import torch
from datasets import Dataset
from tokenizers import Tokenizer, processors
from transformers import PreTrainedTokenizerFast, BatchEncoding

from summarization.config import (
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

    def generate_decoder_targets(self, batch: TEXT) -> BatchEncoding:
        """
        Run __call__ method as a decoder targets tokenizer. Adds END_TOKEN only if decoder input has not been truncated. Final step applies only truncation. Padding is deferred to collator function.
        """
        tokenizer_without_special_token = deepcopy(self._apply_empty_postprocessor())
        tokenizer_with_special_token = deepcopy(self._apply_special_token_postprocessor(END_TOKEN))
        truncation_checker = self._get_truncation_checker()

        batch = [batch] if isinstance(batch, str) else batch
        encodings: list[BatchEncoding] = []
        for text in batch:
            if truncation_checker(text):
                encodings.append(
                    tokenizer_without_special_token(text, max_length=self.max_decoder_steps, truncation=True)
                )
            else:
                encodings.append(tokenizer_with_special_token(text))
        return self._get_batch_encoding_from_list(encodings)

    def prepare_model_inputs(self, batch: dict) -> dict:
        articles, abstracts = batch["article"], batch["highlights"]

        encoding = SummarizationBatchEncoding(
            encoder_inputs=self.generate_encoder_inputs(articles),
            decoder_inputs=self.generate_decoder_inputs(abstracts),
            decoder_targets=self.generate_decoder_targets(abstracts),
        )
        return {**encoding.get_encoder_features(), **encoding.get_decoder_features()}


@dataclass
class SummarizationBatchEncoding:
    encoder_inputs: BatchEncoding
    decoder_inputs: BatchEncoding
    decoder_targets: BatchEncoding

    def get_encoder_features(self) -> dict:
        return {
            "encoder_input_ids": self.encoder_inputs["input_ids"],
            "encoder_padding_mask": self.encoder_inputs["attention_mask"],
        }

    def get_decoder_features(self) -> dict:
        return {
            "decorer_input_ids": self.decoder_inputs["input_ids"],
            "decoder_padding_mask": self.decoder_targets["attention_mask"],
            "decoder_target_ids": self.decoder_targets["input_ids"],
        }


def test_tokenizer():
    tokenizer = SummarizationTokenizerFast.from_pretrained(TOKENIZER_DIR)

    text = "Here is a short article."
    text_2 = "Here is a long article, which exceeds model max length."
    batch = {"article": [text, text_2], "highlights": [text, text_2]}

    features = tokenizer.prepare_model_inputs(batch)
    print(features)


def main():
    test_tokenizer()


if __name__ == "__main__":
    main()

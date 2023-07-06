from typing import Callable, Sequence, Union
from copy import deepcopy

from tokenizers import Tokenizer, processors
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.summarization.config import (
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
        base_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        return cls(
            tokenizer_object=base_tokenizer.backend_tokenizer,
            model_input_names=base_tokenizer.model_input_names,
            model_max_length=base_tokenizer.model_max_length,
            pad_token=base_tokenizer.pad_token,
            unk_token=base_tokenizer.unk_token,
            bos_token=base_tokenizer.bos_token,
            eos_token=base_tokenizer.eos_token,
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

    def get_truncation_checker(self) -> TruncationChecker:
        """
        Truncation checker is used to decide how decoder targets will be encoded. Truncation will be applied if tokens (with added special start token) exceed batch max_decoder_steps.
        """
        checker = self._apply_special_token_postprocessor(START_TOKEN)

        def truncation_checker(single_text: str) -> bool:
            tokens = checker.tokenize(single_text, add_special_tokens=True)
            return len(tokens) > self.max_decoder_steps

        return truncation_checker

    def generate_encoder_inputs(self, text: TEXT) -> list[BatchEncoding]:
        """Run __call__ method as an encoder tokenizer without adding any special tokens."""
        self._apply_empty_postprocessor()
        return self(text, truncation=True, max_length=self.max_encoder_steps)

    def generate_decoder_inputs(self, text: TEXT) -> list[BatchEncoding]:
        """Run __call__ method as a decoder inputs tokenizer. Always adds START_TOKEN."""
        self._apply_special_token_postprocessor(START_TOKEN)
        return self(text, truncation=True, max_length=self.max_decoder_steps)

    def generate_decoder_targets(self, text: TEXT) -> list[BatchEncoding]:
        """
        Run __call__ method as a decoder targets tokenizer. Adds END_TOKEN only if decoder input has not been truncated.
        """
        tokenizer_wo_special_token = deepcopy(self._apply_empty_postprocessor())
        tokenizer_w_special_token = deepcopy(self._apply_special_token_postprocessor(END_TOKEN))
        truncation_checker = self.get_truncation_checker()

        texts = [text] if isinstance(text, str) else text
        tokenized_texts = []
        for text in texts:
            if truncation_checker(text):
                tokenized_texts.append(
                    tokenizer_wo_special_token(text, max_length=self.max_decoder_steps, truncation=True)
                )
            else:
                tokenized_texts.append(
                    tokenizer_w_special_token(text)
                )
        return tokenized_texts


def main():
    tokenizer = SummarizationTokenizerFast.from_pretrained(TOKENIZER_DIR)

    text = "Here is a short article."
    text_2 = "Here is a long article, which exceeds model max length."
    texts = [text, text_2]

    print(tokenizer.generate_encoder_inputs(texts))
    print(tokenizer.generate_decoder_targets(texts))
    print(tokenizer.generate_decoder_inputs(texts))

    # print(target_tokenizer(text_2))

    # model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")
    # tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained("t5-small")

    # inputs = tokenizer(texts, padding=True, truncation=True, max_length=10, return_tensors="pt")
    # labels = inputs["input_ids"]
    # decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
    # tokenizer.model_max_length
    # print(labels)
    # print(decoder_input_ids)


if __name__ == "__main__":
    main()

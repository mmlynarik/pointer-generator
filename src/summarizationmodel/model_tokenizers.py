from pathlib import Path
from typing import Callable, Sequence, Union

from tokenizers import processors

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.configuration_t5 import T5Config
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from src.config import (
    TOKENIZER_DIR,
    START_TOKEN,
    END_TOKEN,
    MAX_DECODER_STEPS,
    MAX_ENCODER_STEPS,
)

TokenizerFunction = Callable[[str], BatchEncoding]
TruncationChecker = Callable[[str], bool]
START_OR_END_TOKEN = Union[START_TOKEN, END_TOKEN]
TEXT = Union[str, Sequence[str]]


def add_start_or_end_token_postprocessor(
    tokenizer: PreTrainedTokenizerFast, token: START_OR_END_TOKEN
) -> PreTrainedTokenizerFast:
    """Append a post-processor to existing tokenizer, adding either [START] or [END] special token."""
    special_token = (token, tokenizer.convert_tokens_to_ids(token))
    template = f"{token}:0 $A:0" if token == START_TOKEN else f"$A:0 {token}:0"
    post_processor = processors.TemplateProcessing(single=template, special_tokens=[special_token])
    tokenizer.backend_tokenizer.post_processor = post_processor
    return tokenizer


def get_truncation_checker(tokenizer_dir: Path, max_length: int) -> TruncationChecker:
    """Truncation checker used to decide which tokenizer will be used for decoder targets."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_start_or_end_token_postprocessor(tokenizer, START_TOKEN)

    def truncation_checker(text: str) -> bool:
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_overflowing_tokens=True)
        return len(inputs["input_ids"]) > 1

    return truncation_checker


def get_encoder_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """Tokenizer used to process encoder inputs without adding any special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def encoder_tokenizer(text: TEXT) -> BatchEncoding:
        return tokenizer(text, truncation=True, padding=True, max_length=max_length)

    return encoder_tokenizer


def get_decoder_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """Tokenizer used to process decoder inputs. Always adds START_TOKEN."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_start_or_end_token_postprocessor(tokenizer, START_TOKEN)

    def decoder_tokenizer(text: TEXT) -> BatchEncoding:
        return tokenizer(text, truncation=True, padding=True, max_length=max_length)

    return decoder_tokenizer


def get_target_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """
    Tokenizer used to process decoder targets. Adds END_TOKEN only if decoder input has not been truncated.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_start_or_end_token_postprocessor(tokenizer, END_TOKEN)

    truncation_checker = get_truncation_checker(tokenizer_dir, max_length)
    encoder_tokenizer = get_encoder_tokenizer(tokenizer_dir, max_length)

    def target_tokenizer(text: TEXT) -> list[BatchEncoding]:
        texts = [text] if isinstance(text, str) else text
        tokenized_texts = []
        for text in texts:
            if truncation_checker(text):
                tokenized_texts.append(encoder_tokenizer(text))
            else:
                tokenized_texts.append(tokenizer(text, truncation=True, padding=True, max_length=max_length))
        return tokenized_texts

    return target_tokenizer


def main():
    encoder_tokenizer = get_encoder_tokenizer(TOKENIZER_DIR, MAX_ENCODER_STEPS)
    decoder_tokenizer = get_decoder_tokenizer(TOKENIZER_DIR, MAX_DECODER_STEPS)
    target_tokenizer = get_target_tokenizer(TOKENIZER_DIR, 8)

    text = "I want you now. Yeah!"
    text_2 = text + "You want me?"
    text_2 = [text, text_2]
    print(encoder_tokenizer(text_2).data)
    print(decoder_tokenizer(text_2).data)
    print(target_tokenizer(text_2))

    # t5_small_moddel = T5ForConditionalGeneration.from_pretrained("t5-small")
    # t5_small_tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained("t5-small")
    # t5_config = T5Config.from_pretrained("t5-small")

    # print(t5_small_tokenizer.pad_token_id, t5_small_tokenizer.pad_token)
    # print(t5_small_tokenizer.unk_token_id, t5_small_tokenizer.unk_token)
    # print(t5_small_tokenizer.eos_token_id, t5_small_tokenizer.eos_token)
    # print(t5_small_tokenizer(text, truncation=True, max_length=6))
    # print(t5_small_tokenizer.model_input_names)
    # print(t5_config.decoder_start_token_id)


if __name__ == "__main__":
    main()

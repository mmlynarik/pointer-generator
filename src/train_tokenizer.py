from pathlib import Path
from typing import Callable, Sequence, Union

from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer, processors
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding

from config import (
    UNK_TOKEN,
    SPECIAL_TOKENS,
    TOKENIZER_DIR,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    VOCAB_SIZE,
    MAX_DECODER_STEPS,
    MAX_ENCODER_STEPS,
)

TokenizerFunction = Callable[[str], BatchEncoding]
TruncationChecker = Callable[[str], bool]
START_OR_END_TOKEN = Union[START_TOKEN, END_TOKEN]
TEXT = Union[str, Sequence[str]]


def add_tokenizer_training_string(batch: dict) -> dict:
    return {"tokenizer_training_string": batch["article"] + " " + batch["highlights"]}


def prepare_cnn_dailymail_dataset() -> Dataset:
    """Loads cnn_dailymail dataset and adds to train split tokenizer training string."""
    dataset = load_dataset("cnn_dailymail", name="3.0.0")
    dataset["train"] = dataset["train"].map(add_tokenizer_training_string)
    return dataset


def train_tokenizer_on_dataset(train_dataset: Dataset, tokenizer_dir: Path):
    """Train lowercased word-level tokenizer on training dataset splitting on whitespace and puctuation."""
    normalizers_list = [
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.NFC(),
        normalizers.Lowercase(),
    ]
    normalizer = normalizers.Sequence(normalizers_list)
    pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    tokenizer = Tokenizer(model=models.WordLevel(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train_from_iterator(train_dataset["tokenizer_training_string"], trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=None,
        padding_side="right",
        truncation_side="right",
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=START_TOKEN,
        eos_token=END_TOKEN,
    )
    tokenizer.save_pretrained(tokenizer_dir)


def get_encoder_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """Tokenizer used to process encoder inputs without adding any special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def encoder_tokenizer(text: TEXT) -> BatchEncoding:
        return tokenizer(
            text, truncation=True, padding=True, max_length=max_length, return_token_type_ids=False
        )

    return encoder_tokenizer


def add_special_token_postprocessor(
    tokenizer: PreTrainedTokenizerFast, token: START_OR_END_TOKEN
) -> PreTrainedTokenizerFast:
    special_token = (token, tokenizer.convert_tokens_to_ids(token))
    template = f"{token}:0 $A:0" if token == START_TOKEN else f"$A:0 {token}:0"
    post_processor = processors.TemplateProcessing(single=template, special_tokens=[special_token])
    tokenizer.backend_tokenizer.post_processor = post_processor
    return tokenizer


def get_truncation_checker(tokenizer_dir: Path, max_length: int) -> TruncationChecker:
    """Truncation checker used to decide which tokenizer to use for decoder targets."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_special_token_postprocessor(tokenizer, START_TOKEN)

    def truncation_checker(text: str) -> bool:
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_overflowing_tokens=True)
        return len(inputs.input_ids) > 1

    return truncation_checker


def get_decoder_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """Tokenizer used to process decoder inputs. Always adds START_TOKEN."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_special_token_postprocessor(tokenizer, START_TOKEN)

    def decoder_tokenizer(text: TEXT) -> BatchEncoding:
        return tokenizer(
            text, truncation=True, padding=True, max_length=max_length, return_token_type_ids=False
        )

    return decoder_tokenizer


def get_target_tokenizer(tokenizer_dir: Path, max_length: int) -> TokenizerFunction:
    """
    Tokenizer used to process decoder targets. Adds END_TOKEN only if decoder input has not been truncated.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = add_special_token_postprocessor(tokenizer, END_TOKEN)

    truncation_checker = get_truncation_checker(tokenizer_dir, max_length)
    encoder_tokenizer = get_encoder_tokenizer(tokenizer_dir, max_length)

    def target_tokenizer(text: TEXT) -> BatchEncoding:
        texts = [text] if isinstance(text, str) else text
        tokenized_texts = []
        for text in texts:
            if truncation_checker(text):
                tokenized_texts.append(encoder_tokenizer(text))
            else:
                tokenized_texts.append(
                    tokenizer(
                        text,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_token_type_ids=False,
                    )
                )
        return tokenized_texts

    return target_tokenizer


def main():
    # dataset = prepare_cnn_dailymail_dataset()
    # train_tokenizer_on_dataset(dataset["train"], TOKENIZER_DIR)

    encoder_tokenizer = get_encoder_tokenizer(TOKENIZER_DIR, MAX_ENCODER_STEPS)
    decoder_tokenizer = get_decoder_tokenizer(TOKENIZER_DIR, MAX_DECODER_STEPS)
    target_tokenizer = get_target_tokenizer(TOKENIZER_DIR, 8)

    text = "I want you now. Yeah!"
    text_2 = text + "You want me?"
    text_2 = [text, text_2]
    print(encoder_tokenizer(text_2).data)
    print(decoder_tokenizer(text_2).encodings)
    print(target_tokenizer(text_2)[0].encodings)


if __name__ == "__main__":
    main()

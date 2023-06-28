from pathlib import Path

from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding

from config import (
    UNK_TOKEN,
    SPECIAL_TOKENS,
    TOKENIZER_DIR,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    VOCAB_SIZE,
)


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


def main():
    dataset = prepare_cnn_dailymail_dataset()
    train_tokenizer_on_dataset(dataset["train"], TOKENIZER_DIR)


if __name__ == "__main__":
    main()

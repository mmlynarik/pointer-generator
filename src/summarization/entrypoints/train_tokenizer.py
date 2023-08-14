from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from summarization.datamodule.config import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    TOKENIZER_DIR,
    UNK_TOKEN,
    VOCAB_SIZE,
    MODEL_MAX_LENGTH,
)
from summarization.datamodule.dataset import load_cnn_dailymail_dataset


def train_base_tokenizer_on_dataset(train_dataset: Dataset, tokenizer_dir: Path):
    """
    Train base lowercased word-level tokenizer on training dataset splitting on whitespace and puctuation.
    Actual tokenization of texts will be performed using specific tokenizers with added postprocessors.
    """

    normalizers_list = [
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.NFC(),
        normalizers.Lowercase(),
    ]
    normalizer = normalizers.Sequence(normalizers_list)
    pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
    trainer = trainers.WordLevelTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    backend_tokenizer = Tokenizer(model=models.WordLevel(unk_token=UNK_TOKEN))
    backend_tokenizer.normalizer = normalizer
    backend_tokenizer.pre_tokenizer = pre_tokenizer
    backend_tokenizer.train_from_iterator(train_dataset["tokenizer_training_string"], trainer=trainer)
    backend_tokenizer.enable_padding(
        pad_id=backend_tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN, length=None
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_input_names=["input_ids", "attention_mask"],  # for the underlying tokenizer to work correctly
        model_max_length=MODEL_MAX_LENGTH,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=START_TOKEN,
        eos_token=END_TOKEN,
    )
    tokenizer.save_pretrained(tokenizer_dir)


def main():
    dataset = load_cnn_dailymail_dataset(version="3.0.0")
    train_base_tokenizer_on_dataset(dataset["train"], TOKENIZER_DIR)


if __name__ == "__main__":
    main()

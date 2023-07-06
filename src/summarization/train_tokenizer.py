from pathlib import Path

from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.summarization.datamodule.dataset import prepare_cnn_dailymail_dataset
from src.summarization.config import (
    UNK_TOKEN,
    SPECIAL_TOKENS,
    TOKENIZER_DIR,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    VOCAB_SIZE,
)


def train_base_tokenizer_on_dataset(train_dataset: Dataset, tokenizer_dir: Path):
    """
    Train base lowercased word-level tokenizer on training dataset splitting on whitespace and puctuation.
    Actual tokenization of texts will be applied using specific postprocessor-enriched tokenizers.
    """

    normalizers_list = [
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.NFC(),
        normalizers.Lowercase(),
    ]
    normalizer = normalizers.Sequence(normalizers_list)
    pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    backend_tokenizer = Tokenizer(model=models.WordLevel(unk_token=UNK_TOKEN))
    backend_tokenizer.normalizer = normalizer
    backend_tokenizer.pre_tokenizer = pre_tokenizer
    backend_tokenizer.train_from_iterator(train_dataset["tokenizer_training_string"], trainer=trainer)
    backend_tokenizer.enable_padding(pad_id=backend_tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN, length=None)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_input_names=["input_ids", "attention_mask", "aj toto tu chcem este"],
        model_max_length=1024,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=START_TOKEN,
        eos_token=END_TOKEN,
    )
    tokenizer.save_pretrained(tokenizer_dir)


def main():
    dataset = prepare_cnn_dailymail_dataset()
    train_base_tokenizer_on_dataset(dataset["train"], TOKENIZER_DIR)


if __name__ == "__main__":
    main()

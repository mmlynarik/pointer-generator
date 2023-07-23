from datasets import Dataset, load_dataset


def add_tokenizer_training_string(batch: dict) -> dict:
    return {"tokenizer_training_string": batch["article"] + " " + batch["highlights"]}


def load_cnn_dailymail_dataset(version: str) -> Dataset:
    """Loads cnn_dailymail dataset and adds to train split tokenizer training string."""
    dataset = load_dataset("cnn_dailymail", name=version)
    dataset["train"] = dataset["train"].map(add_tokenizer_training_string)
    return dataset

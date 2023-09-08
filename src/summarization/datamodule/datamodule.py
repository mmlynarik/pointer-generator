import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_from_disk
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from summarization.datamodule.config import DATA_DIR, TOKENIZER_DIR
from summarization.datamodule.dataset import load_cnn_dailymail_dataset
from summarization.datamodule.tokenizer import SummarizationTokenizerFast

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass
class SummarizationDataCollator:
    tokenizer: SummarizationTokenizerFast
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def pad_one_feature(self, batch: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        items = [example[name] for example in batch]
        max_length = max(len(i) for i in items)
        for example in batch:
            example[name] = torch.tensor(
                example[name] + [self.tokenizer.pad_token_id] * (max_length - len(example[name])),
                dtype=torch.int32,
            )

        return batch

    def __call__(self, batch: list[dict[str, Any]]):
        for feature_name in self.tokenizer.paddable_features:
            batch = self.pad_one_feature(batch, feature_name)

        paddable_features = {
            key: torch.stack([example[key].clone() for example in batch])
            for key in batch[0].keys()
            if key in self.tokenizer.paddable_features
        }
        non_paddable_features = {
            key: [example[key] for example in batch]
            for key in batch[0].keys()
            if key in self.tokenizer.non_paddable_features
        }
        return {**paddable_features, **non_paddable_features}


class SummarizationDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_version: str = "3.0.0",
        tokenizer_dir: Path = TOKENIZER_DIR,
        data_dir: Path = DATA_DIR,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_version = dataset_version
        self.tokenizer_dir = tokenizer_dir
        self.data_dir = data_dir
        self.tokenizer = SummarizationTokenizerFast.from_pretrained(tokenizer_dir)

    def prepare_data(self) -> None:
        if Path(self.data_dir / "dataset_dict.json").exists():
            log.info(f"CNN & DailyMail dataset already exists in {self.data_dir} folder.")
            return

        raw_dataset = load_cnn_dailymail_dataset(version=self.dataset_version)
        encoded_dataset = raw_dataset.map(
            self.tokenizer.prepare_model_inputs,
            batched=True,
            batch_size=1000,
            remove_columns=["article", "highlights", "id"],
        )
        encoded_dataset["train"] = encoded_dataset["train"].remove_columns("tokenizer_training_string")
        encoded_dataset.save_to_disk(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        encoded_dataset = load_from_disk(self.data_dir)
        self.data_collator = SummarizationDataCollator(self.tokenizer)

        if stage is None or stage == "fit":
            self.train_dataset = encoded_dataset["train"]
        if stage is None or stage in ["fit", "validate"]:
            self.val_dataset = encoded_dataset["validation"]
        if stage is None or stage == "test":
            self.test_dataset = encoded_dataset["test"]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = SummarizationDataModule(batch_size=16)
    dm.prepare_data()
    dm.setup()

    for step, batch in enumerate(dm.val_dataloader()):
        if step == 0:
            print(batch["encoder_input_ids"])
            break

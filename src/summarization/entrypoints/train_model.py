from lightning.pytorch.cli import LightningCLI

from summarization.datamodule.datamodule import SummarizationDataModule
from summarization.model.model import AbstractiveSummarizationModel


def main():
    cli = LightningCLI(AbstractiveSummarizationModel, SummarizationDataModule, save_config_callback=None)


if __name__ == "__main__":
    main()

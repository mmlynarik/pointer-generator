from lightning.pytorch.cli import LightningCLI

from summarizationmodel.model import AbstractiveSummarizationModel
from summarizationmodel.datamodule.datamodule import SummarizationDataModule


def main():
    cli = LightningCLI(AbstractiveSummarizationModel, SummarizationDataModule)


if __name__ == "__main__":
    main()

from lightning.pytorch.cli import LightningCLI

from summarizationmodel.datamodule.datamodule import SummarizationDataModule
from summarizationmodel.model.model import AbstractiveSummarizationModel


def main():
    cli = LightningCLI(AbstractiveSummarizationModel, SummarizationDataModule)


if __name__ == "__main__":
    main()

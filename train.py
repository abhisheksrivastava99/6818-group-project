"""Train BankSimFM models and persist demo artifacts."""

from banksimfm.models.training import train_models


if __name__ == "__main__":
    artifacts = train_models()
    print(artifacts.metrics)

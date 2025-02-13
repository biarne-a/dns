import argparse
import sys

from dns.train.config import Config
from dns.train.run_training import run_training


def _parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_dir", required=False, type=str, default="gs://ml-25m")
    # data_dir can be different than gcs_dir if for example the data has been downloaded on local storage
    parser.add_argument("--data_dir", required=False, type=str, default="gs://ml-25m")
    parser.add_argument("--loss_type", required=False, type=str, default="sm")
    parser.add_argument("--nb_epochs", required=False, type=int, default=1)
    parser.add_argument("--batch_size", required=False, type=int, default=64)
    parser.add_argument("--nb_levels", required=False, type=int, default=4)
    parser.add_argument("--embedding_dimension", required=False, type=int, default=128)
    return Config(vars(parser.parse_args(args=sys.argv[1:])))


def run():
    config = _parse_config()
    run_training(config)


if __name__ == "__main__":
    run()

import re
from typing import Dict


class Config:
    def __init__(self, args: Dict):
        self._args = args
        self.data_dir = args.get("data_dir")
        self.gcs_dir = args.get("gcs_dir", "")
        self.loss_type = args.get("loss_type")
        self.nb_epochs = args.get("nb_epochs")
        self.batch_size = args.get("batch_size")
        self.embedding_dimension = args.get("embedding_dimension")
        self.bucket_name = self._extract_bucket_name()

    def _extract_bucket_name(self):
        match = re.match("^gs://(.+)$", self.gcs_dir)
        if not match:
            raise Exception("gcs_dir must start with gs://")
        return match.groups()[0]

    @property
    def exp_name(self):
        return f"{self.loss_type}_{self.embedding_dimension}"

    def to_json(self):
        return self._args

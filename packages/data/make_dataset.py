import copy
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import CamembertTokenizer

from ..utilities.text_helper import text_preprocessor

DATASET_NAME = "wisesight_sentiment"


class SequenceData:
    def __init__(
        self,
        do_train: bool,
        do_val: bool,
        do_test: bool,
        max_seq_length: int,
        tokenizer: CamembertTokenizer,
    ) -> None:
        self.dataset = load_dataset(DATASET_NAME)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if do_train:
            self.preprocessed_train_dataset = self._preprocess_dataset("train")
        if do_val:
            self.preprocessed_train_dataset = self._preprocess_dataset("validation")
        if do_test:
            self.preprocessed_train_dataset = self._preprocess_dataset("test")

    def get_dataset(self, split_type: str):
        if split_type == "train":
            return self.train_dataset
        elif split_type == "val":
            return self.validation_dataset
        elif split_type == "test":
            return self.test_dataset
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def _preprocess_dataset(self, split_type: str) -> Dict[str, List[int]]:
        _dataset = copy.deepcopy(self.dataset)
        _dataset = _dataset[split_type].add_column(
            "preprocessed_texts",
            [
                text_preprocessor(text)
                for text in tqdm(
                    _dataset[split_type]["texts"],
                    desc=f"Preprocessing {split_type} dataset",
                )
            ],
        )
        preprocessed_dataset = self.tokenizer(
            _dataset["preprocessed_texts"],
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        preprocessed_dataset["label_ids"] = self.dataset[split_type]["category"]

        return preprocessed_dataset

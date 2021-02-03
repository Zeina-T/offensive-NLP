import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 2
MAX_LEN = 512


class OffensiveDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_data_loader(
    data,
    model_name,
    cased,
):
    train_tweets, train_labels, test_tweets, test_labels = data
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_tweets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
    )

    test_encodings = tokenizer(
        test_tweets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
    )

    training_set = OffensiveDataset(train_encodings, train_labels)
    testing_set = OffensiveDataset(test_encodings, test_labels)

    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}

    test_params = {"batch_size": TEST_BATCH_SIZE, "shuffle": True, "num_workers": 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader

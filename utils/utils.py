import json
import random
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

DATA_DIR = Path("data")
DATA_LANGUAGES = ["gr", "ar", "tr", "da", "en"]
random.seed(123)


def read_data(json_path):
    data = json.load(open(json_path, "r"))
    for key, lang_dict in data.items():
        for k, v in lang_dict.items():
            data[key][k] = pd.read_json(v)
    return data


DATA = read_data(DATA_DIR / "data_2021-02-02_23-55.json")


def preprocess_data(language_code, cased=True, number_samples=None, equal_sample=True):
    assert language_code in DATA_LANGUAGES

    train_data = DATA[language_code]["train"]
    train_data["label"] = pd.get_dummies(train_data["label"], drop_first=True)

    test_data = DATA[language_code]["test"]
    test_data["label"] = pd.get_dummies(test_data["label"], drop_first=True)

    if not cased:
        train_data["tweet"] = train_data["tweet"].apply(lambda x: x.lower())
        test_data["tweet"] = test_data["tweet"].apply(lambda x: x.lower())

    if number_samples is not None and equal_sample:
        samples_per_class = number_samples // len(train_data.label.unique())
        train_data = (
            train_data.groupby("label")
            .apply(lambda x: x.sample(samples_per_class))
            .reset_index(drop=True)
        )  # sample equally from each class
        train_data = train_data.sample(number_samples)  # shuffle

        train_tweets = train_data["tweet"].values.tolist()
        train_labels = train_data["label"].values.tolist()
        test_tweets = test_data["tweet"].values.tolist()
        test_labels = test_data["label"].values.tolist()
    elif number_samples is not None and not equal_sample:  # random sampling
        train_data = train_data.sample(n=number_samples)

        train_tweets = train_data["tweet"].values.tolist()
        train_labels = train_data["label"].values.tolist()
        test_tweets = test_data["tweet"].values.tolist()
        test_labels = test_data["label"].values.tolist()

    else:  # if not using number_samples sample the data to avoid unbalanced classes
        max_sample = train_data["label"].value_counts().max()
        class_size = {i: max_sample for i in train_data["label"].unique()}

        sampler = RandomOverSampler(sampling_strategy=class_size)

        train_X, train_y = sampler.fit_resample(
            train_data[["tweet"]], train_data["label"]
        )
        test_X, test_y = sampler.fit_resample(test_data[["tweet"]], test_data["label"])

        train_tweets = train_X.values.tolist()
        test_tweets = test_X.values.tolist()

        train_labels = train_y.values.tolist()
        test_labels = test_y.values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def get_combined_language_data(
    base_language, low_language=None, cased=True, number_low_samples=None
):
    assert base_language in DATA_LANGUAGES
    if low_language:
        assert low_language in DATA_LANGUAGES
        assert isinstance(number_low_samples, int)

    base_tweets, base_labels, _, _ = preprocess_data(base_language, cased)
    (
        low_train_tweets,
        low_train_labels,
        low_test_tweets,
        low_test_labels,
    ) = preprocess_data(low_language, cased, number_low_samples)

    combined_train = list(zip(base_tweets, base_labels)) + list(
        zip(low_train_tweets, low_train_labels)
    )
    random.shuffle(combined_train)
    combined_train_tweets = [i[0] for i in combined_train]
    combined_train_labels = [i[1] for i in combined_train]

    return (
        combined_train_tweets,
        combined_train_labels,
        low_test_tweets,
        low_test_labels,
    )


if __name__ == "__main__":
    preprocess_data("tr", cased=True, number_samples=None)

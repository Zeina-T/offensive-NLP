import json
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AdamW, BertForSequenceClassification

from loops import train, validate
from utils.dataloaders import get_data_loader
from utils.metrics import calculate_scores
from utils.utils import get_combined_language_data, preprocess_data

random.seed(123)
monolingual_models = {
    "gr": ("nlpaueb/bert-base-greek-uncased-v1", False),
    "ar": ("asafaya/bert-base-arabic", False),
    "tr": ("dbmdz/bert-base-turkish-cased", True),
    "da": ("DJSammy/bert-base-danish-uncased_BotXO,ai", False),
    "en": ("bert-base-cased", True),
}

multi_lingual_model = "bert-base-cased"  # "bert-base-multilingual-cased"
BASE_EPOCHS = 20
LOW_EPOCHS = 10
LEARNING_RATE = 1e-01
DECAY_FACTOR = 0.3
RESULTS_DIR = Path("results")
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def separate_few_shot_experiment(base_lang, low_lang, sample_count):
    exp_id = random.randint(0, 1e6)

    base_tweets, base_labels, _, _ = preprocess_data(base_language, cased=True)
    low_data = preprocess_data(
        low_language, cased=True, number_samples=number_low_samples, equal_sample=True
    )

    base_training_loader = get_data_loader(
        (base_tweets, base_labels),
        multi_lingual_model,
        cased=False,
        testing_loader=False,
    )
    low_training_loader, low_testing_loader = get_data_loader(
        low_data,
        multi_lingual_model,
        cased=False,
        testing_loader=True,
    )

    model = BertForSequenceClassification.from_pretrained(multi_lingual_model)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_FACTOR)

    for epoch in range(EPOCHS):
        print(f"---- BASE EPOCH: {epoch} ----")
        train(model, base_training_loader, epoch, optimizer, loss_function)

    for epoch in range(EPOCHS):
        print(f"---- LOW EPOCH: {epoch} ----")
        train(model, low_training_loader, epoch, optimizer, loss_function)

    validate_target, validate_predictions = validate(
        model, low_testing_loader, loss_function
    )

    torch.save(
        validate_target,
        RESULTS_DIR
        / f"raw_data/few_shot_{base_lang}_{low_lang}_{sample_count}_{exp_id}_target.pt",
    )
    torch.save(
        validate_predictions,
        RESULTS_DIR
        / f"raw_data/few_shot_{base_lang}_{low_lang}_{sample_count}_{exp_id}_predictions.pt",
    )

    scores = calculate_scores(validate_target, validate_predictions)

    save_experiment(
        exp_id,
        base_lang,
        low_lang,
        sample_count,
        multi_lingual_model,
        LEARNING_RATE,
        EPOCHS,
        scores,
    )


def combined_few_shot_experiment(base_lang, low_lang, sample_count):
    exp_id = random.randint(0, 1e6)

    train_test_data = get_combined_language_data(
        base_lang, low_lang, cased=True, number_low_samples=sample_count
    )

    training_loader, testing_loader = get_data_loader(
        train_test_data,
        multi_lingual_model,
        cased=False,
    )

    model = BertForSequenceClassification.from_pretrained(multi_lingual_model)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train(model, training_loader, epoch, optimizer, loss_function)

    validate_target, validate_predictions = validate(
        model, testing_loader, loss_function
    )

    torch.save(
        validate_target,
        RESULTS_DIR
        / f"raw_data/few_shot_{base_lang}_{low_lang}_{sample_count}_{exp_id}_target.pt",
    )
    torch.save(
        validate_predictions,
        RESULTS_DIR
        / f"raw_data/few_shot_{base_lang}_{low_lang}_{sample_count}_{exp_id}_predictions.pt",
    )

    scores = calculate_scores(validate_target, validate_predictions)

    save_experiment(
        exp_id,
        base_lang,
        low_lang,
        sample_count,
        multi_lingual_model,
        LEARNING_RATE,
        EPOCHS,
        scores,
    )


def save_experiment(
    exp_id, base_lang, low_lang, sample_count, model_name, lr, epoch, scores
):
    result = {
        "id": exp_id,
        "base_lang": base_lang,
        "low_lang": low_lang,
        "sample_count": sample_count,
        "model_name": model_name,
        "learning_rate": lr,
        "epochs": epoch,
        "accuracy": scores[0],
        "precision": scores[1],
        "recall": scores[2],
        "f1": scores[3],
        "_time": datetime.today().strftime("%Y-%m-%d_%H-%M"),
    }
    filename = f"exp_{exp_id}_{base_lang}_{low_lang}_{sample_count}"
    if (RESULTS_DIR / f"{filename}.json").exists():
        filename += f"_{random.randint(0, 100)}"
    json.dump(result, open(RESULTS_DIR / f"{filename}.json", "w"))


if __name__ == "__main__":
    samples = [10, 50, 100, 200, 500]

    for i in samples:
        separate_few_shot_experiment("en", "tr", i)

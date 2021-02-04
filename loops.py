import torch

from utils.dataloaders import get_data_loader
from utils.metrics import calcuate_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def train(model, training_loader, epoch, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in enumerate(training_loader, 0):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs.logits, targets.flatten())
        tr_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1).flatten()

        n_correct += calcuate_accuracy(preds, targets.flatten())
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 1000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"--- Training Loss per 1000 steps: {loss_step}")
            print(f"--- Training Accuracy per 1000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


def validate(model, testing_loader, loss_function):
    model.eval()
    n_correct = 0
    tst_loss = 0
    nb_tst_steps = 0
    nb_tst_examples = 0

    epoch_preds = torch.Tensor().to(device)
    epoch_targets = torch.Tensor().to(device)
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            loss = loss_function(outputs.logits, targets.flatten())
            tst_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).flatten()

            n_correct += calcuate_accuracy(preds, targets.flatten())
            nb_tst_steps += 1
            nb_tst_examples += targets.size(0)

            if _ % 1000 == 0:
                loss_step = tst_loss / nb_tst_steps
                accu_step = (n_correct * 100) / nb_tst_examples
                print(f"--- Validation Loss per 100 steps: {loss_step}")
                print(f"--- Validation Accuracy per 100 steps: {accu_step}")

            epoch_preds = torch.cat((epoch_preds, preds), dim=0)
            epoch_targets = torch.cat((epoch_targets, targets.flatten()), dim=0)

    return epoch_targets, epoch_preds


if __name__ == "__main__":
    model_name = "bert-base-uncased"
    training_loader, testing_loader = get_data_loader(
        "en",
        model_name,
        cased=False,
    )
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-05)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        train(model, training_loader, epoch)

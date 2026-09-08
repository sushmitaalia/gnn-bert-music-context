import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score


def evaluate_task1(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= 0.5).int()

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_labels.append(
                batch["labels"].numpy()
            )

    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)

    micro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return micro_f1, macro_f1


def evaluate_task2_gnn(
    gnn_model,
    test_gnn_loader,
    device
):
    gnn_model.eval()

    correct = 0
    total = 0

    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for batch in test_gnn_loader:
            batch = batch.to(device)

            logits = gnn_model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == batch.y
            ).sum().item()

            total += batch.y.size(0)

            all_test_predictions.extend(
                predictions.cpu().numpy()
            )

            all_test_labels.extend(
                batch.y.cpu().numpy()
            )

    test_accuracy = correct / total

    return test_accuracy


def evaluate_task2_cnn(
    cnn_model,
    test_cnn_loader,
    device
):
    cnn_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_cnn_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = cnn_model(inputs)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    cnn_test_accuracy = correct / total

    return cnn_test_accuracy


def evaluate_task3(
    model,
    val_task3_loader,
    test_task3_loader,
    device
):
    model.eval()

    val_probs = []
    val_true = []

    with torch.no_grad():
        for batch in val_task3_loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                graph,
                input_ids,
                attention_mask
            )

            val_probs.append(
                torch.sigmoid(logits).cpu().numpy()
            )

            val_true.append(
                batch["labels"].numpy()
            )

    val_probs = np.vstack(val_probs)
    val_true = np.vstack(val_true)

    best_threshold = 0.5
    best_macro = -1

    for threshold in np.arange(
        0.05,
        0.51,
        0.01
    ):
        preds = (
            val_probs >= threshold
        ).astype(int)

        macro = f1_score(
            val_true,
            preds,
            average="macro",
            zero_division=0
        )

        if macro > best_macro:
            best_macro = macro
            best_threshold = threshold

    test_probs = []
    test_true = []

    with torch.no_grad():
        for batch in test_task3_loader:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(
                graph,
                input_ids,
                attention_mask
            )

            test_probs.append(
                torch.sigmoid(logits).cpu().numpy()
            )

            test_true.append(
                batch["labels"].numpy()
            )

    test_probs = np.vstack(test_probs)
    test_true = np.vstack(test_true)

    test_preds = (
        test_probs >= best_threshold
    ).astype(int)

    micro = f1_score(
        test_true,
        test_preds,
        average="micro",
        zero_division=0
    )

    macro = f1_score(
        test_true,
        test_preds,
        average="macro",
        zero_division=0
    )

    auc_pr = average_precision_score(
        test_true,
        test_probs,
        average="macro"
    )

    return (
        best_threshold,
        micro,
        macro,
        auc_pr
    )


def recall_at_k(similarity_matrix, k):
    correct = 0

    for i in range(similarity_matrix.size(0)):
        topk = torch.topk(
            similarity_matrix[i],
            k
        ).indices

        if i in topk:
            correct += 1

    return correct / similarity_matrix.size(0)

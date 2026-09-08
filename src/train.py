import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_task1_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_graphsage(
    gnn_model,
    train_gnn_loader,
    val_gnn_loader,
    device,
    num_epochs=20
):
    criterion = nn.BCEWithLogitsLoss()

    gnn_optimizer = torch.optim.Adam(
        gnn_model.parameters(),
        lr=0.001
    )

    gnn_history = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": []
    }

    for epoch in range(1, num_epochs + 1):
        gnn_model.train()
        total_loss = 0

        for batch in train_gnn_loader:
            batch = batch.to(device)

            gnn_optimizer.zero_grad()

            logits = gnn_model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            targets = F.one_hot(
                batch.y,
                num_classes=8
            ).float()

            loss = criterion(
                logits,
                targets
            )

            loss.backward()
            gnn_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_gnn_loader)

        gnn_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_gnn_loader:
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

        val_accuracy = correct / total

        gnn_history["epoch"].append(epoch)
        gnn_history["train_loss"].append(avg_loss)
        gnn_history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    return gnn_history


def train_cnn(
    cnn_model,
    train_cnn_loader,
    val_cnn_loader,
    device,
    num_epochs=10
):
    cnn_criterion = nn.CrossEntropyLoss()

    cnn_optimizer = torch.optim.Adam(
        cnn_model.parameters(),
        lr=0.001
    )

    cnn_history = {
        "epoch": [],
        "train_loss": [],
        "val_accuracy": []
    }

    for epoch in range(1, num_epochs + 1):
        cnn_model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_cnn_loader,
            desc=f"Epoch {epoch}"
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            cnn_optimizer.zero_grad()

            outputs = cnn_model(inputs)

            loss = cnn_criterion(
                outputs,
                labels
            )

            loss.backward()
            cnn_optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_cnn_loader)

        cnn_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_cnn_loader:
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

        val_accuracy = correct / total

        cnn_history["epoch"].append(epoch)
        cnn_history["train_loss"].append(avg_loss)
        cnn_history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    return cnn_history


def train_task3_model(
    model,
    train_task3_loader,
    device,
    model_name,
    epochs=3
):
    criterion = nn.BCEWithLogitsLoss()

    lr = 1e-3 if model_name == "gnn" else 2e-5

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        progress = tqdm(
            train_task3_loader,
            desc=f"{model_name} Epoch {epoch}"
        )

        for batch in progress:
            graph = batch["graph"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(
                graph,
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch} loss:",
            round(
                total_loss / len(train_task3_loader),
                4
            )
        )


def train_task4(
    model,
    train_loader,
    device,
    contrastive_loss,
    epochs=5
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for graphs, input_ids, attention_mask, ytids in train_loader:
            graphs = graphs.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()

            audio_embedding, text_embedding = model(
                graphs,
                input_ids,
                attention_mask
            )

            loss = contrastive_loss(
                audio_embedding,
                text_embedding
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {avg_loss:.4f}"
        )

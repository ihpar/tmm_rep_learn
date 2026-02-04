import torch
from dataset import build_dataloaders
from bilstm import BiLSTMAttentionClassifier


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "f_names"}

        optimizer.zero_grad()

        logits, _ = model(batch)

        loss = criterion(logits, batch["labels"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "f_names"}

        logits, _ = model(batch)

        loss = criterion(logits, batch["labels"])

        total_loss += loss.item() * batch["labels"].size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=100,
    patience=7,
    save_path="best_model.pt"
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                },
                save_path
            )
            print(f"Best model saved (val_loss={best_val_loss:.4f})")

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement "
                f"({epochs_without_improvement}/{patience})"
            )

            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionClassifier(
        pc_vocab=9,
        acc_vocab=10,
        oct_vocab=7,
        dur_vocab=46,
        meas_vocab=4
    )
    model.to(device)

    train_loader, class_weights, val_loader, _ = build_dataloaders(
        "vocab.pkl", "dataset.pkl", batch_size=4)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    num_epochs = 1

    for epoch in range(1, num_epochs + 1):
        print(f"train epoch {epoch}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print("eval...")
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()

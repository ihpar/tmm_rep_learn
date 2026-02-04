import torch
import pickle
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def collate_fn(batch):
    # sort the batch w.r.t. piece lengths, desc
    batch.sort(key=lambda x: x["length"], reverse=True)

    pcs = [b["pcs"] for b in batch]
    accs = [b["accs"] for b in batch]
    octs = [b["octs"] for b in batch]
    durs = [b["durs"] for b in batch]
    meas = [b["meas"] for b in batch]

    labels = torch.stack([b["label"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch])
    f_names = np.array([b["f_name"] for b in batch])

    pcs_pad = pad_sequence(pcs, batch_first=True, padding_value=0)
    accs_pad = pad_sequence(accs, batch_first=True, padding_value=0)
    octs_pad = pad_sequence(octs, batch_first=True, padding_value=0)
    durs_pad = pad_sequence(durs, batch_first=True, padding_value=0)
    meas_pad = pad_sequence(meas, batch_first=True, padding_value=0)

    # Mask: 1 = real token, 0 = padding (fake token)
    mask = (pcs_pad != 0).long()

    return {
        "pcs": pcs_pad,
        "accs": accs_pad,
        "octs": octs_pad,
        "durs": durs_pad,
        "meas": meas_pad,
        "mask": mask,
        "lengths": lengths,
        "labels": labels,
        "f_names": f_names
    }


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        weights.append(total / (num_classes * counts[c]))

    return torch.tensor(weights, dtype=torch.float)


class MakamDataset(Dataset):
    def __init__(self, pieces, labels):
        """
        pieces: List[ {"f_name": str, "notes": List[Tuple[int,int,int,int,int]]} ]
                Each piece = [(pc, acc, oct, dur, meas), ...]
        labels: List[int]  (makam labels 0-9)
        """
        assert len(pieces) == len(labels)

        self.pieces = pieces
        self.labels = labels

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        piece = self.pieces[idx]
        label = self.labels[idx]

        # Split features into separate tensors
        pcs = torch.tensor([n[0] for n in piece["notes"]], dtype=torch.long)
        accs = torch.tensor([n[1] for n in piece["notes"]], dtype=torch.long)
        octs = torch.tensor([n[2] for n in piece["notes"]], dtype=torch.long)
        durs = torch.tensor([n[3] for n in piece["notes"]], dtype=torch.long)
        meas = torch.tensor([n[4] for n in piece["notes"]], dtype=torch.long)

        return {
            "pcs": pcs,
            "accs": accs,
            "octs": octs,
            "durs": durs,
            "meas": meas,
            "label": torch.tensor(label, dtype=torch.long),
            "length": len(piece["notes"]),
            "f_name": piece["f_name"]
        }


def build_dataloaders(vocab_file, dataset_file, batch_size=16, rs=42):
    """
    returns train_loader, val_loader, test_loader
    """

    vocab, dataset = None, None
    with open(vocab_file, "rb") as in_file:
        vocab = pickle.load(in_file)
    with open(dataset_file, "rb") as in_file:
        dataset = pickle.load(in_file)

    pieces, labels = [], []
    for makam, makam_pieces in dataset.items():
        label = vocab["makam"][makam]
        for makam_piece in makam_pieces:
            labels.append(label)
            pieces.append(makam_piece)

    X_train, X_rest, y_train, y_rest = train_test_split(
        pieces, labels, test_size=0.3, random_state=rs)

    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=rs)

    ds_train = MakamDataset(X_train, y_train)
    train_labels = [label for label in y_train]
    class_weights = compute_class_weights(train_labels, num_classes=12)
    ds_val = MakamDataset(X_val, y_val)
    ds_test = MakamDataset(X_test, y_test)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, class_weights, val_loader, test_loader


def main():
    train_loader, class_weights, val_loader, test_loader = build_dataloaders(
        "vocab.pkl", "dataset.pkl", batch_size=4)
    print(len(train_loader))
    print(class_weights)
    print(len(val_loader))
    print(len(test_loader))

    loaders = [train_loader, val_loader, test_loader]
    for loader in loaders:
        batch = next(iter(loader))
        print(batch["pcs"].shape,
              batch["accs"].shape,
              batch["octs"].shape,
              batch["durs"].shape,
              batch["meas"].shape,
              batch["mask"].shape,
              batch["lengths"].shape,
              batch["labels"],
              batch["f_names"]
              )
        print("-" * 50)


if __name__ == "__main__":
    main()

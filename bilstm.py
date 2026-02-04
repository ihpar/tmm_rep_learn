import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import build_dataloaders


class MaskedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, mask):
        """
        lstm_out: (B: batch size, T: time steps, H: hidden dim)
        mask:     (B, T)   1 = valid (real), 0 = padding (fake)
        """
        # (B, T, 1)
        scores = self.attn(lstm_out)

        # Mask padding tokens
        scores = scores.squeeze(-1)          # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)   # (B, T)

        # Weighted sum
        context = torch.sum(
            lstm_out * weights.unsqueeze(-1),
            dim=1
        )  # (B, H)

        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        pc_vocab,
        acc_vocab,
        oct_vocab,
        dur_vocab,
        meas_vocab,
        emb_dim=32,
        lstm_hidden=128,
        num_classes=12,
        dropout=0.3
    ):
        super().__init__()

        self.pc_emb = nn.Embedding(pc_vocab, emb_dim, padding_idx=0)
        self.acc_emb = nn.Embedding(acc_vocab, emb_dim, padding_idx=0)
        self.oct_emb = nn.Embedding(oct_vocab, emb_dim, padding_idx=0)
        self.dur_emb = nn.Embedding(dur_vocab, emb_dim, padding_idx=0)
        self.meas_emb = nn.Embedding(meas_vocab, emb_dim, padding_idx=0)

        total_emb_dim = emb_dim * 5

        self.bilstm = nn.LSTM(
            input_size=total_emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.attention = MaskedAttention(hidden_dim=2 * lstm_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, batch):
        pcs = batch["pcs"]
        accs = batch["accs"]
        octs = batch["octs"]
        durs = batch["durs"]
        meas = batch["meas"]
        mask = batch["mask"]

        pc_e = self.pc_emb(pcs)
        acc_e = self.acc_emb(accs)
        oct_e = self.oct_emb(octs)
        dur_e = self.dur_emb(durs)
        meas_e = self.meas_emb(meas)

        x = torch.cat([pc_e, acc_e, oct_e, dur_e, meas_e], dim=-1)

        lstm_out, _ = self.bilstm(x)

        context, attn_weights = self.attention(lstm_out, mask)

        logits = self.classifier(context)

        return logits, attn_weights


def main():
    train_loader, _, _ = build_dataloaders("vocab.pkl", "dataset.pkl", batch_size=4)
    model = BiLSTMAttentionClassifier(
        pc_vocab=9,
        acc_vocab=10,
        oct_vocab=7,
        dur_vocab=46,
        meas_vocab=4
    )

    batch = next(iter(train_loader))
    print(batch["pcs"].shape)
    logits, attn_weights = model(batch)
    print(logits.size())
    print(attn_weights.size())
    print(attn_weights.sum(dim=1))
    print((attn_weights * (1 - batch["mask"])).max())


if __name__ == "__main__":
    main()

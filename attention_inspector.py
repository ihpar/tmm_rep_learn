import torch
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal
from scipy.stats import spearmanr
from collections import defaultdict
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class AttentionInspector:
    def __init__(self, device, model, dataloader):
        self.device = device
        self.model = model
        self.dataloader = dataloader
        # makam
        self.makam_vocab = {
            "hicaz": 0, "nihavent": 1, "ussak": 2, "rast": 3,
            "huzzam": 4, "segah": 5, "huseyni": 6, "mahur": 7,
            "hicazkar": 8, "kurdilihicazkar": 9, "muhayyer": 10, "saba": 11
        }
        self.makam_vocab_inv = {v: k for k, v in self.makam_vocab.items()}
        # PC
        self.pc_vocab = {
            "PAD": 0, "Rest": 1,
            "G": 2, "A": 3, "B": 4, "C": 5, "D": 6, "E": 7, "F": 8
        }
        self.pc_vocab_inv = {v: k for k, v in self.pc_vocab.items()}
        # Acc
        self.acc_vocab = {"PAD": 0, "": 1,
                          "#1": 2, "#2": 3, "#3": 4, "#4": 5, "#5": 6, "#6": 7, "#7": 8, "#8": 9}
        self.acc_vocab_inv = {v: k for k, v in self.acc_vocab.items()}
        # Dur
        self.dur_vocab = {"PAD": 0,
                          0.007812: 1, 0.008929: 2, 0.010417: 3, 0.015625: 4, 0.017857: 5,
                          0.020833: 6, 0.025: 7, 0.027778: 8, 0.03125: 9, 0.035714: 10,
                          0.041667: 11, 0.05: 12, 0.055556: 13, 0.0625: 14, 0.071429: 15,
                          0.083333: 16, 0.1: 17, 0.111111: 18, 0.125: 19, 0.142857: 20,
                          0.166667: 21, 0.1875: 22, 0.2: 23, 0.222222: 24, 0.25: 25,
                          0.3: 26, 0.333333: 27, 0.375: 28, 0.4: 29, 0.5: 30, 0.666667: 31, 1.0: 32}
        self.dur_vocab_inv = {v: k for k, v in self.dur_vocab.items()}

    @torch.no_grad()
    def get_predictions(self):
        self.model.eval()

        (self.all_labels,
         self.all_preds,
         self.all_f_names,
         self.all_attn_weights,
         self.all_true_lengths,
         self.all_pcs,
         self.all_accs,
         self.all_meas,
         self.all_durs) = ([], [], [], [], [], [], [], [], [])

        for batch in self.dataloader:
            f_names = batch["f_names"]
            lengths = batch["lengths"]
            batch["accs"]
            batch["meas"]
            batch = {k: v.to(self.device)
                     for k, v in batch.items() if k != "f_names"}
            logits, attn_weights = self.model(batch)
            labels = batch["labels"]
            preds = torch.argmax(logits, dim=1)

            self.all_preds.append(preds.cpu().numpy())
            self.all_labels.append(labels.cpu().numpy())
            self.all_f_names.append(f_names)
            self.all_attn_weights.extend(attn_weights.cpu().numpy())
            self.all_true_lengths.append(lengths.cpu().numpy())
            self.all_pcs.extend(batch["pcs"].cpu().numpy())
            self.all_accs.extend(batch["accs"].cpu().numpy())
            self.all_meas.extend(batch["meas"].cpu().numpy())
            self.all_durs.extend(batch["durs"].cpu().numpy())

        self.all_preds = np.concatenate(self.all_preds)
        self.all_labels = np.concatenate(self.all_labels)
        self.all_f_names = np.concatenate(self.all_f_names)
        self.all_true_lengths = np.concatenate(self.all_true_lengths)

        return (
            self.all_labels,
            self.all_preds,
            self.all_f_names,
            self.all_attn_weights,
            self.all_true_lengths,
            self.all_pcs,
            self.all_accs,
            self.all_meas,
            self.all_durs
        )

    def export_vocabs(self):
        return (
            self.makam_vocab,
            self.makam_vocab_inv,
            self.pc_vocab,
            self.pc_vocab_inv,
            self.acc_vocab,
            self.acc_vocab_inv,
            self.dur_vocab,
            self.dur_vocab_inv
        )

    def evaluate_classification(self, y_true, y_pred, makam_names):
        results = {}

        # Overall accuracy
        results["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None
        )

        # Macro & weighted
        results["macro_f1"] = np.mean(f1)
        results["weighted_f1"] = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted"
        )[2]

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        report = classification_report(
            y_true,
            y_pred,
            target_names=makam_names,
            digits=4,
            output_dict=True
        )
        results["report"] = report
        return results

    def get_single_piece_ai(self, piece_idx):
        # all_labels, all_preds, all_f_names, all_attn_weights, all_true_lengths
        true_len = self.all_true_lengths[piece_idx]

        attn_piece = self.all_attn_weights[piece_idx][:true_len]
        pcs_piece = self.all_pcs[piece_idx][:true_len]
        accs_piece = self.all_accs[piece_idx][:true_len]
        meas_piece = self.all_meas[piece_idx][:true_len]
        durs_piece = self.all_durs[piece_idx][:true_len]
        return (
            attn_piece,
            pcs_piece,
            accs_piece,
            meas_piece,
            durs_piece
        )

    def combine_pitch_pc_acc(self, pc_idx, acc_idx):
        pc = self.pc_vocab_inv.get(pc_idx, "UNK")
        acc_name = self.acc_vocab_inv.get(acc_idx, "natural")

        if pc == "PAD":
            return "PAD"
        if pc == "Rest":
            return "Rest"

        return f"{pc}{acc_name}"

    def aggregate_attention_by_pitch(self, attn, pitch_ids):
        """
        attn: (L,)
        pcs:  (L,) pitch class ids
        """
        agg = defaultdict(float)
        for a, p in zip(attn, pitch_ids):
            agg[p] += a
        return agg

    def aggregate_attention_by_pitch_acc(self, attn, pcs, accs):
        """
        attn: (L,)
        pcs:  (L,) pitch class ids
        accs: (L,) accidental ids
        """
        agg = defaultdict(float)

        for a, pc, acc in zip(attn, pcs, accs):
            if pc == 0:  # PAD
                continue
            agg[(pc, acc)] += float(a)

        return agg

    def normalize_attention(self, agg_dict):
        total = sum(agg_dict.values())
        return {k: v / total for k, v in agg_dict.items()}

    def aggregate_attention_pitch_acc_duration(self, attn, pcs, accs, durs):
        agg = defaultdict(float)

        for a, pc, acc, dur in zip(attn, pcs, accs, durs):
            if pc == 0:  # PAD
                continue
            dur_ratio = self.dur_vocab_inv[dur]
            agg[(pc, acc)] += float(a * dur_ratio)

        return agg

    def attention_duration_correlation(self, attn, durs):
        attn_arr = []
        dur_arr = []
        for a, d in zip(attn, durs):
            dur_ratio = self.dur_vocab_inv[d]
            dur_arr.append(dur_ratio)
            attn_arr.append(a)

        return spearmanr(attn, durs).correlation

    def get_makam_pieces(self, makam_id, correct=True):
        """
        {'hicaz': 0, 'nihavent': 1, 'ussak': 2, 'rast': 3,
        'huzzam': 4, 'segah': 5, 'huseyni': 6, 'mahur': 7,
        'hicazkar': 8, 'kurdilihicazkar': 9, 'muhayyer': 10, 'saba': 11}
        """
        res = []
        for i, (y_hat, y) in enumerate(zip(self.all_preds, self.all_labels)):
            if y != makam_id:
                continue
            if correct and y_hat == y:
                res.append(i)
            if not correct and y_hat != y:
                res.append(i)
        return np.array(res)

    def aggregate_attention_for_makam(self, makam_idx):
        num_pieces = len(self.all_pcs)
        makam_agg = defaultdict(float)
        piece_count = 0

        for i in range(num_pieces):
            if self.all_labels[i] != makam_idx:
                continue

            true_len = self.all_true_lengths[i]
            attn_piece = self.all_attn_weights[i][:true_len]
            pcs_piece = self.all_pcs[i][:true_len]
            accs_piece = self.all_accs[i][:true_len]

            piece_agg = self.aggregate_attention_by_pitch_acc(
                attn_piece, pcs_piece, accs_piece
            )

            for k, v in piece_agg.items():
                makam_agg[k] += v

            piece_count += 1

        return makam_agg, piece_count

    def aggregate_attention_correct_vs_incorrect(self, makam_idx):
        agg_correct = defaultdict(float)
        agg_incorrect = defaultdict(float)

        count_correct = 0
        count_incorrect = 0

        num_pieces = len(self.all_pcs)
        for i in range(num_pieces):
            if self.all_labels[i] != makam_idx:
                continue

            true_len = self.all_true_lengths[i]
            attn_piece = self.all_attn_weights[i][:true_len]
            pcs_piece = self.all_pcs[i][:true_len]
            accs_piece = self.all_accs[i][:true_len]

            piece_agg = self.aggregate_attention_by_pitch_acc(
                attn_piece, pcs_piece, accs_piece
            )

            if self.all_preds[i] == self.all_labels[i]:
                for k, v in piece_agg.items():
                    agg_correct[k] += v
                count_correct += 1
            else:
                for k, v in piece_agg.items():
                    agg_incorrect[k] += v
                count_incorrect += 1

        return agg_correct, agg_incorrect, count_correct, count_incorrect

    def attention_entropy(self, attn_dict):
        p = np.array(list(attn_dict.values()))
        return -np.sum(p * np.log(p + 1e-9))

    def get_mean_attn_dur_correlation_makam(self, makam):
        corrs = []
        makam_idx = self.get_makam_pieces(self.makam_vocab[makam])
        for idx in makam_idx:
            true_len_i = self.all_true_lengths[idx]
            attn_piece_i = self.all_attn_weights[idx][:true_len_i]
            durs_piece_i = self.all_durs[idx][:true_len_i]
            corr_i = self.attention_duration_correlation(
                attn_piece_i, durs_piece_i)
            corrs.append(corr_i)

        return np.array(corrs).mean()

    def get_mean_attn_dur_correlation_all(self):
        corrs = []
        num_pieces = len(self.all_pcs)
        for idx in range(num_pieces):
            true_len_i = self.all_true_lengths[idx]
            attn_piece_i = self.all_attn_weights[idx][:true_len_i]
            durs_piece_i = self.all_durs[idx][:true_len_i]
            corr_i = self.attention_duration_correlation(
                attn_piece_i, durs_piece_i)
            corrs.append(corr_i)
        return np.array(corrs).mean()

    def collect_attention_duration(self, target_makam=None):
        """
        Returns a DataFrame with columns:
        ['duration', 'attention', 'makam']
        """
        records = []
        num_pieces = len(self.all_pcs)
        for i in range(num_pieces):
            makam_id = self.all_labels[i]
            makam_name = self.makam_vocab_inv[makam_id]

            if (target_makam is not None) and (makam_name != target_makam):
                continue

            T = len(self.all_durs[i])
            for t in range(T):
                if t == self.all_true_lengths[i]:
                    break

                dur_id = self.all_durs[i][t]
                if dur_id == 0:
                    continue  # PAD

                dur_value = self.dur_vocab_inv[dur_id]
                if dur_value == 0:
                    continue  # ornaments

                attn = self.all_attn_weights[i][t]

                records.append({
                    "duration": dur_value,
                    "attention": attn,
                    "makam": makam_name
                })

        df = pd.DataFrame(records)
        df_binned = self.rebin_relative_duration(df)
        return df_binned

    def rebin_relative_duration(self, df):
        bins = [0.0, 0.0625, 0.1875, 0.375, 0.75, 1.01]
        labels = [
            "Very short (≤1/16)",
            "Short (≤3/16)",
            "Medium (≤3/8)",
            "Long (≤3/4)",
            "Very long (>3/4)"
        ]

        df["dur_bin"] = pd.cut(
            df["duration"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        return df

    def cliffs_delta(self, x, y):
        nx, ny = len(x), len(y)
        greater = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return (greater - less) / (nx * ny)

    def prob_high_attention(self, attentions, threshold):
        return np.mean(np.array(attentions) > threshold)

    def kruskal_wallis(self, attn_by_bin):
        groups = [vals for vals in attn_by_bin.values() if len(vals) > 0]
        H, p = kruskal(*groups)
        return H, p

    def dunn_posthoc(self, attn_by_bin):
        data = []
        for bin_name, vals in attn_by_bin.items():
            for v in vals:
                data.append({"duration_bin": bin_name, "attention": v})

        df_ph = pd.DataFrame(data)

        # Dunn post-hoc with Holm correction
        df_dunn = sp.posthoc_dunn(
            df_ph,
            val_col="attention",
            group_col="duration_bin",
            p_adjust="holm"
        )
        return df_dunn

    def chi_sqaure(self, attn_by_bin, df_attn, bin_x, bin_y):
        threshold = np.percentile(df_attn, 95)

        for bin_name, vals in attn_by_bin.items():
            print(bin_name, self.prob_high_attention(vals, threshold))

        short = attn_by_bin[bin_x]
        long = attn_by_bin[bin_y]

        table = [
            [sum(np.array(short) > threshold), sum(
                np.array(short) <= threshold)],
            [sum(np.array(long) > threshold),  sum(np.array(long) <= threshold)]
        ]

        chi2, p, _, _ = chi2_contingency(table)
        return chi2, p

    def collect_attn_dur_bin_stats(
            self,
            bin_x="Very short (≤1/16)",
            bin_y="Very long (>3/4)",
            target_makam=None
    ):
        """
        bins:

        Very short (≤1/16) 
        Short (≤3/16)
        Medium (≤3/8)
        Long (≤3/4)
        Very long (>3/4)

        returns: binned DataFrame
        """

        df = self.collect_attention_duration(target_makam)

        # kruskal
        attention_by_bin = (
            df
            .groupby("dur_bin", observed=False)["attention"]
            .apply(list)
            .to_dict()
        )

        H, p = self.kruskal_wallis(attention_by_bin)
        print(f"Kruskal-Wallis H = {H:.3f}, p-value = {p:.4e}")

        print("\n" + "- " * 30 + "\n")

        # Dunn post hoc
        df_dunn = self.dunn_posthoc(attention_by_bin)
        print("Dunn post-hoc with Holm correction")
        print(df_dunn)
        df_dunn.to_excel("dunn.xlsx")

        print("\n" + "- " * 30 + "\n")

        # cliff's delta
        delta = self.cliffs_delta(
            attention_by_bin[bin_x],
            attention_by_bin[bin_y]
        )

        print(f"Cliff's delta = {delta:.3f}")

        print("\n" + "- " * 30 + "\n")

        # Chi²
        chi2, p = self.chi_sqaure(
            attention_by_bin,
            df["attention"].to_numpy(),
            bin_x,
            bin_y
        )
        print(f"Chi² = {chi2:.3f}, p = {p:.4e}")

        return df


def main():
    pass


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt
from attention_inspector import AttentionInspector


class PlotManager:
    def __init__(self):
        pass

    def plot_confusion_matrix(
        self,
        cm,
        class_names,
        title="Normalized Confusion Matrix",
        save_fig=False
    ):
        # Normalize row-wise
        cm = cm.astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(7, 6))
        sns.set_theme(style="white", font="Times")

        ax = sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Proportion"},
            annot_kws={"size": 10}
        )

        ax.set_xlabel("Predicted Makam", fontsize=12)
        ax.set_ylabel("True Makam", fontsize=12)
        ax.figure.axes[-1].yaxis.label.set_size(12)
        ax.set_title(title, fontsize=12, pad=12)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        plt.tight_layout()
        if save_fig:
            plt.savefig("cm.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_attention_over_time(
            self,
            attn,
            title="Attention over Notes",
            save_fig=False
    ):
        plt.figure(figsize=(6, 3))
        plt.plot(attn, linewidth=2)
        plt.xlabel("Note Index", fontname="Times", fontsize=12)
        plt.ylabel("Attention Weight", fontname="Times", fontsize=12)
        plt.title(title, fontname="Times", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_fig:
            plt.savefig("attn_time.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_attention_with_pitch(
            self,
            attn,
            pcs,
            pc_vocab_inv,
            save_fig=False
    ):
        pitches = [pc_vocab_inv[p] for p in pcs]

        plt.figure(figsize=(24, 3))
        plt.bar(range(len(attn)), attn)
        plt.xticks(
            range(len(pitches)),
            pitches,
            rotation=90,
            fontsize=6,
            fontname="Times"
        )
        plt.xlabel("Note Index", fontname="Times", fontsize=12)
        plt.ylabel("Attention Weight", fontname="Times", fontsize=12)
        plt.title("Attention Aligned with Pitch Classes",
                  fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("pitch_attn.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_attention_with_measures(
            self,
            attn,
            meas,
            save_fig=False
    ):
        plt.figure(figsize=(12, 3))
        plt.plot(attn, linewidth=2)

        for i, m in enumerate(meas):
            if m == 3:  # measure end {'PAD': 0, 'start': 1, 'middle': 2, 'end': 3}
                plt.axvline(i, color="red", alpha=0.2)

        plt.xlabel("Note Index", fontname="Times", fontsize=12)
        plt.ylabel("Attention Weight", fontname="Times", fontsize=12)
        plt.title("Attention with Measure Boundaries",
                  fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("meas_attn.pdf", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_attention_with_combined_pitch(
        self,
        attn,
        pcs,
        accs,
        ai: AttentionInspector,
        save_fig=False
    ):
        pitch_labels = [
            ai.combine_pitch_pc_acc(p, a)
            for p, a in zip(pcs, accs)
        ]

        plt.figure(figsize=(24, 3))
        plt.bar(range(len(attn)), attn)
        plt.xticks(
            range(len(pitch_labels)),
            pitch_labels,
            rotation=90,
            fontsize=6,
            fontname="Times"
        )
        plt.xlabel("Note Index", fontname="Times", fontsize=12)
        plt.ylabel("Attention Weight", fontname="Times", fontsize=12)
        plt.title("Attention Aligned with Combined Pitch Classes",
                  fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("comb_pitch_attn.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_agg_attention_by_pitch(
            self,
            agg,
            ai: AttentionInspector,
            save_fig=False
    ):
        labels = []
        values = []

        for (pc, acc), val in sorted(agg.items(), key=lambda x: -x[1]):
            pc_name = ai.pc_vocab_inv[pc]
            acc_name = ai.acc_vocab_inv[acc]
            labels.append(f"{pc_name}{acc_name}")
            values.append(val)

        plt.figure(figsize=(6, 3))
        plt.bar(labels, values)
        plt.xticks(rotation=45, fontname="Times", fontsize=10)
        plt.ylabel("Total Attention", fontname="Times", fontsize=12)
        plt.title("Aggregated Attention by Pitch + Accidental",
                  fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("agg_attn_by_pitch.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_agg_attention_by_pitch_duration(
            self,
            agg,
            ai: AttentionInspector,
            save_fig=False
    ):
        labels = []
        values = []

        for (pc, acc), val in sorted(agg.items(), key=lambda x: -x[1]):
            pc_name = ai.pc_vocab_inv[pc]
            acc_name = ai.acc_vocab_inv[acc]
            labels.append(f"{pc_name}{acc_name}")
            values.append(val)

        plt.figure(figsize=(6, 3))
        plt.bar(labels, values)
        plt.xticks(rotation=45, fontname="Times", fontsize=10)
        plt.ylabel("Total Attention", fontname="Times", fontsize=12)
        plt.title("Duration Weighted Aggregated Attention by Pitch + Accidental",
                  fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("agg_attn_by_pitch_dur.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_agg_attention_correct_vs_incorrect(
            self,
            makam,
            ai: AttentionInspector,
            save_fig=False
    ):
        agg_c, agg_i, n_c, n_i = ai.aggregate_attention_correct_vs_incorrect(
            ai.makam_vocab[makam])

        agg_c = ai.normalize_attention(agg_c)
        agg_i = ai.normalize_attention(agg_i)

        rows = []

        for key in set(agg_c.keys()).union(agg_i.keys()):
            pc, acc = key
            label = f"{ai.pc_vocab_inv[pc]}{ai.acc_vocab_inv[acc]}"

            rows.append({
                "Pitch": label,
                "Attention": agg_c.get(key, 0),
                "Type": "Correct"
            })
            rows.append({
                "Pitch": label,
                "Attention": agg_i.get(key, 0),
                "Type": "Incorrect"
            })
        df = pd.DataFrame(rows)
        plt.figure(figsize=(7, 3))
        sns.barplot(data=df, x="Pitch", y="Attention", hue="Type")
        plt.xticks(rotation=45, fontname="Times", fontsize=10)
        plt.title(
            f"Attention by Pitch+Accidental: "
            f"Correct vs Incorrect ({makam.capitalize()})",
            fontname="Times", fontsize=12
        )
        plt.tight_layout()
        if save_fig:
            plt.savefig("attn_by_pitch_corr_incorr.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

        entropy_correct = ai.attention_entropy(agg_c)
        entropy_incorrect = ai.attention_entropy(agg_i)

        print("Entropy (Correct):", entropy_correct)
        print("Entropy (Incorrect):", entropy_incorrect)

    def plot_agg_attention_by_pitch_duration_makam(
            self,
            makam,
            ai: AttentionInspector,
            save_fig=False
    ):
        makam_agg, piece_count = ai.aggregate_attention_for_makam(
            ai.makam_vocab[makam])

        makam_agg = ai.normalize_attention(makam_agg)
        labels = []
        values = []

        for (pc, acc), val in sorted(
            makam_agg.items(), key=lambda x: -x[1]
        ):
            labels.append(f"{ai.pc_vocab_inv[pc]}{ai.acc_vocab_inv[acc]}")
            values.append(val)

        plt.figure(figsize=(6, 3))
        plt.bar(labels, values)
        plt.xticks(rotation=45, fontname="Times", fontsize=10)
        plt.title(
            f"Aggregated Attention by Pitch + Accidental "
            f"({makam.capitalize()})",
            fontname="Times", fontsize=12
        )
        plt.ylabel("Normalized Attention", fontname="Times", fontsize=12)
        plt.tight_layout()
        if save_fig:
            plt.savefig("attn_by_pitch_corr_incorr.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_attention_vs_duration_boxplot(
            self,
            df,
            title=None,
            save_fig=False
    ):
        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="dur_bin",
            y="attention",
            color="skyblue",
            showfliers=False
        )

        # sns.stripplot(
        #     data=df,
        #     x="dur_bin",
        #     y="attention",
        #     color="black",
        #     size=2,
        #     alpha=0.25,
        #     jitter=True
        # )

        plt.xlabel("Relative duration (binned)",
                   fontname="Times", fontsize=12)
        plt.ylabel("Attention weight",
                   fontname="Times", fontsize=12)
        plt.title("Attention vs Relative Duration (Binned)",
                  fontname="Times", fontsize=12)
        plt.xticks(fontname="Times", fontsize=10)

        plt.tight_layout()
        if save_fig:
            plt.savefig("attn_vs_dur_box.pdf",
                        bbox_inches="tight", pad_inches=0.05)
        plt.show()

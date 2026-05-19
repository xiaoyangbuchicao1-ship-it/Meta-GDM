import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(series: pd.Series, window: int = 50) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def plot_metric(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str,
                output_path: Path, color: str, smooth_window: int) -> None:
    data = pd.to_numeric(df[y_col], errors="coerce")
    x = pd.to_numeric(df[x_col], errors="coerce")
    valid = ~(x.isna() | data.isna())
    x = x[valid]
    data = data[valid]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, data, color=color, alpha=0.18, linewidth=0.8, label="原始值")
    ax.plot(x, moving_average(data, smooth_window), color=color, linewidth=2.4,
            label=f"{smooth_window}轮移动平均")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("训练轮次", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_curves(training_csv: Path, output_dir: Path, smooth_window: int = 50) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(training_csv)

    metrics = [
        ("Success", "共识达成情况", "Success", "success_rate.png", "#22b8cf"),
        ("Final Consensus", "最终共识水平", "Final Consensus", "final_consensus.png", "#ff7f0e"),
        ("Episode Return", "训练回合回报", "Episode Return", "episode_return.png", "#2ca02c"),
        ("Actor Loss", "Actor损失", "Actor Loss", "actor_loss.png", "#1f77b4"),
        ("Critic Loss", "Critic损失", "Critic Loss", "critic_loss.png", "#d62728"),
    ]

    for column, title, ylabel, filename, color in metrics:
        if column in df.columns:
            plot_metric(
                df,
                x_col="Episode",
                y_col=column,
                title=title,
                ylabel=ylabel,
                output_path=output_dir / filename,
                color=color,
                smooth_window=smooth_window,
            )

    print(f"训练曲线已保存至：{output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Meta-GDM main training curves.")
    parser.add_argument("--training-csv", required=True, help="训练历史 CSV 路径。")
    parser.add_argument("--output-dir", default="results/figures/training_curves", help="曲线图输出目录。")
    parser.add_argument("--smooth-window", type=int, default=50, help="移动平均窗口大小。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_training_curves(
        training_csv=Path(args.training_csv),
        output_dir=Path(args.output_dir),
        smooth_window=args.smooth_window,
    )

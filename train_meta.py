import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from agent import MetaAgent
from context_utils import append_context_step
from env import OpinionDynamicsEnv
from plot_training_curves import plot_training_curves
from replay_buffer import MetaReplayBuffer


RESULT_DIR = Path("results")
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"


def resolve_device(device_arg="auto"):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 torch.cuda.is_available() 为 False。")
    return device_arg


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = values - values.min()
    if np.allclose(values.sum(), 0.0):
        return 0.0
    values = np.sort(values)
    index = np.arange(1, values.size + 1)
    return float((2 * np.sum(index * values) / (values.size * np.sum(values))) - (values.size + 1) / values.size)


def make_env(args, num_agents, seed):
    return OpinionDynamicsEnv(
        num_agents=num_agents,
        max_steps=args.max_steps,
        consensus_threshold=args.threshold,
        seed=seed,
        opinion_range=tuple(args.opinion_range),
        stubbornness_range=tuple(args.stubbornness_range),
        cost_sensitivity_range=tuple(args.cost_range),
    )


def train_main_model(args):
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(args, args.train_agents, args.seed)
    agent = MetaAgent(
        z_dim=args.z_dim,
        context_window=args.context_window,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        kl_weight=args.kl_weight,
        num_heads=args.num_heads,
        device=device,
    )
    replay_buffer = MetaReplayBuffer(capacity=args.buffer_capacity, context_window=agent.context_window)
    warmup_episodes = min(args.max_warmup_episodes, max(10, int(args.train_episodes * args.warmup_fraction)))

    curriculum = []
    for stage_threshold, step_frac in zip(args.curriculum_thresholds, args.curriculum_step_fracs):
        curriculum.append((min(stage_threshold, args.threshold), max(5, int(args.max_steps * step_frac))))
    curriculum[-1] = (args.threshold, args.max_steps)
    stage_len = max(1, args.train_episodes // len(curriculum))

    training_records = []
    progress = tqdm(range(args.train_episodes), desc="train Meta-GDM main", dynamic_ncols=True)
    for episode in progress:
        stage_idx = min(episode // stage_len, len(curriculum) - 1)
        env.consensus_threshold, env.max_steps = curriculum[stage_idx]
        state = env.reset()
        context = np.zeros((args.train_agents, agent.context_window, 3))
        episode_data = []
        episode_reward = 0.0
        update_stats = {"critic_loss": np.nan, "actor_loss": np.nan, "kl_loss": np.nan}
        info = {
            "consensus_level": env._calculate_consensus_level_from_opinions(env.opinions),
            "success": False
        }

        for step in range(env.max_steps):
            if episode < warmup_episodes:
                action = np.random.uniform(-1.0, 1.0, args.train_agents)
            else:
                exploration_noise = max(0.2 * (1.0 - episode / max(args.train_episodes, 1)), 0.03)
                action = agent.select_action(
                    state,
                    context,
                    evaluate=False,
                    exploration_noise=exploration_noise
                )

            next_state, reward, done, info = env.step(action)
            episode_data.append((state, action, reward, next_state, float(not done), info))
            episode_reward += float(np.mean(reward))
            context = append_context_step(
                context,
                info["suggestions"],
                info["actual_movements"],
                reward
            )
            state = next_state
            if done:
                break

        replay_buffer.push_episode(episode_data)
        if episode >= warmup_episodes and len(replay_buffer) >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size, device=device)
            update_stats = agent.update(batch)

        training_records.append({
            "Episode": episode + 1,
            "Stage": stage_idx + 1,
            "Curriculum Threshold": env.consensus_threshold,
            "Curriculum Max Steps": env.max_steps,
            "Steps": step + 1,
            "Episode Return": episode_reward,
            "Final Consensus": float(info.get("consensus_level", 0.0)),
            "Success": int(bool(info.get("success", False))),
            "Critic Loss": float(update_stats.get("critic_loss", np.nan)),
            "Actor Loss": float(update_stats.get("actor_loss", np.nan)),
            "KL Loss": float(update_stats.get("kl_loss", np.nan)),
        })

        if episode % 10 == 0 or episode == args.train_episodes - 1:
            progress.set_postfix({
                "cons": f"{info.get('consensus_level', 0.0):.3f}",
                "success": int(info.get("success", False)),
                "thr": f"{env.consensus_threshold:.2f}",
                "steps": step + 1,
                "reward": f"{episode_reward:.1f}",
                "buffer": len(replay_buffer)
            })

    env_tag = hashlib.md5(json.dumps(env.get_reward_parameters(), sort_keys=True).encode("utf-8")).hexdigest()[:8]
    lr_tag = f"{args.lr:g}".replace(".", "p").replace("-", "m")
    suffix = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    checkpoint_path = CHECKPOINT_DIR / (
        f"meta_gdm_main_z{args.z_dim}_cw{args.context_window}_h{args.num_heads}_"
        f"lr{lr_tag}_ep{args.train_episodes}_seed{args.seed}_{env_tag}{suffix}.pth"
    )
    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_training_history.csv")
    pd.DataFrame(training_records).to_csv(history_path, index=False, encoding="utf-8-sig")
    agent.save_checkpoint(
        checkpoint_path,
        episode=args.train_episodes,
        extra={
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": args.alpha,
            "warmup_fraction": args.warmup_fraction,
            "curriculum": curriculum,
            "seed": args.seed,
            "reward_parameters": env.get_reward_parameters(),
            "training_history_csv": str(history_path),
        }
    )
    print(f"\n训练历史已保存：{history_path}")
    print(f"模型 checkpoint 已保存：{checkpoint_path}")
    if not args.no_plot:
        plot_training_curves(
            training_csv=history_path,
            output_dir=RESULT_DIR / "figures" / "training_curves",
            smooth_window=args.smooth_window,
        )
    return checkpoint_path


def evaluate_checkpoint(args, checkpoint_path):
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = MetaAgent(
        z_dim=int(checkpoint.get("z_dim", args.z_dim)),
        context_window=int(checkpoint.get("context_window", args.context_window)),
        hidden_dim=args.hidden_dim,
        num_heads=int(checkpoint.get("num_heads", args.num_heads)),
        device=device,
    )
    agent.load_checkpoint(checkpoint_path)
    agent.set_eval_mode()

    records = []
    try:
        for num_agents in args.eval_agents:
            progress = tqdm(range(args.eval_episodes), desc=f"eval N={num_agents}", dynamic_ncols=True)
            for ep in progress:
                env_seed = args.seed + ep * 997 + num_agents * 13
                env = make_env(args, num_agents, env_seed)
                state = env.reset()
                context = np.zeros((num_agents, agent.context_window, 3))
                cumulative_costs = np.zeros(num_agents)
                boundary_violations = 0
                episode_return = 0.0
                success = False
                steps = args.max_steps
                info = {"consensus_level": env._calculate_consensus_level_from_opinions(env.opinions)}

                for step in range(args.max_steps):
                    action = agent.select_action(state, context, evaluate=True)
                    next_state, reward, done, info = env.step(action)
                    step_costs = env.calculate_adjustment_costs(info["actual_movements"])
                    cumulative_costs += step_costs
                    boundary_violations += int(np.sum((env.opinions < env.safe_low) | (env.opinions > env.safe_high)))
                    episode_return += float(np.mean(reward))
                    context = append_context_step(
                        context,
                        info["suggestions"],
                        info["actual_movements"],
                        reward
                    )
                    state = next_state
                    if done:
                        success = info["success"]
                        steps = step + 1
                        break

                records.append({
                    "Model": "Meta-GDM Main",
                    "Agents": num_agents,
                    "Episode": ep,
                    "Success": int(success),
                    "Steps": steps,
                    "Total Cost": float(np.sum(cumulative_costs)),
                    "Cost Gini": gini(cumulative_costs),
                    "Boundary Violations": boundary_violations,
                    "Episode Return": episode_return,
                    "Final Consensus": float(info.get("consensus_level", 0.0)),
                })
    finally:
        agent.set_train_mode()

    raw = pd.DataFrame(records)
    raw_path = RESULT_DIR / "meta_gdm_main_eval_raw.csv"
    summary_path = RESULT_DIR / "meta_gdm_main_eval_summary.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary = raw.groupby(["Model", "Agents"]).agg({
        "Success": "mean",
        "Steps": "mean",
        "Total Cost": "mean",
        "Cost Gini": "mean",
        "Boundary Violations": "mean",
        "Episode Return": "mean",
        "Final Consensus": "mean",
    }).reset_index()
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n评估汇总：")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\n评估结果已保存：\n  {raw_path}\n  {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the main Meta-GDM model only.")
    parser.add_argument("--train-agents", type=int, default=10)
    parser.add_argument("--eval-agents", nargs="+", type=int, default=[10, 40, 100])
    parser.add_argument("--train-episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--opinion-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--stubbornness-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--cost-range", nargs=2, type=float, default=[0.1, 0.9])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--context-window", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--max-warmup-episodes", type=int, default=200)
    parser.add_argument("--curriculum-thresholds", nargs=4, type=float, default=[0.70, 0.80, 0.85, 0.90])
    parser.add_argument("--curriculum-step-fracs", nargs=4, type=float, default=[0.75, 5.0 / 6.0, 11.0 / 12.0, 1.00])
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument("--skip-eval", action="store_true", help="只训练并保存模型，不进行训练后评估。")
    parser.add_argument("--no-plot", action="store_true", help="训练结束后不自动绘制训练曲线。")
    parser.add_argument("--smooth-window", type=int, default=50, help="训练曲线移动平均窗口大小。")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    ckpt = train_main_model(cli_args)
    if not cli_args.skip_eval:
        evaluate_checkpoint(cli_args, ckpt)

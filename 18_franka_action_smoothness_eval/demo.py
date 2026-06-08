"""评估 Franka 动作轨迹平滑性的离线工具。

这个脚本不依赖 Isaac Sim，可以直接读取 `17_franka_smolvla_data_collection`
保存出来的 `.npz` 轨迹，对每条 episode 做平滑性体检。

它重点回答两个问题：

1. 这条数据的动作标签本身是不是过于生硬？
2. 这条数据在执行后的末端轨迹是不是已经出现明显抖动？

输出内容包括：

1. 每条 episode 的平滑性分数
2. 动作轨迹与执行轨迹的速度 / 加速度 / jerk 指标
3. 方向频繁反转、夹爪来回切换等抖动迹象
4. 一个 CSV 报表和一个 JSON 汇总文件

运行示例：

    python isaac-sim-learning-demos/18_franka_action_smoothness_eval/demo.py
    python isaac-sim-learning-demos/18_franka_action_smoothness_eval/demo.py --episode 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "17_franka_smolvla_data_collection"
    / "outputs"
    / "raw"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
FALLBACK_OUTPUT_DIR = Path("/tmp/isaac_action_smoothness_reports")
DEFAULT_DT = 1.0 / 20.0


def parse_args() -> argparse.Namespace:
    """解析少量必要参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="待评估的 episode 目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="评估报表输出目录。",
    )
    parser.add_argument("--episode", type=int, default=None, help="只评估某一个 episode。")
    return parser.parse_args()


def summarize_values(values: np.ndarray) -> dict[str, float]:
    """返回均值 / P95 / 最大值。"""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def time_axis(length: int, dt: float, start_at_one: bool = False) -> np.ndarray:
    """构造时间轴。

    差分后的量长度会更短，这里统一用对应采样点的时间坐标去画。
    """

    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    offset = 1 if start_at_one else 0
    return (np.arange(length, dtype=np.float64) + offset) * dt


def finite_difference_norms(sequence: np.ndarray, order: int, dt: float) -> np.ndarray:
    """计算位置序列的一阶/二阶/三阶差分范数。

    - `order=1` 表示速度近似
    - `order=2` 表示加速度近似
    - `order=3` 表示 jerk 近似
    """

    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.shape[0] <= order:
        return np.zeros(0, dtype=np.float64)
    diff = np.diff(sequence, n=order, axis=0) / (dt**order)
    return np.linalg.norm(diff, axis=1)


def step_norms(sequence: np.ndarray) -> np.ndarray:
    """计算逐帧位移长度。"""

    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.shape[0] <= 1:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(sequence, axis=0), axis=1)


def direction_change_metrics(sequence: np.ndarray, active_step_threshold: float = 1e-4) -> dict[str, float]:
    """评估轨迹是否频繁反向或急转。

    这里忽略几乎不动的微小步长，避免把正常停顿误判成抖动。
    """

    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.shape[0] <= 2:
        return {"reverse_ratio": 0.0, "hard_turn_ratio": 0.0}

    steps = np.diff(sequence, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    active_indices = np.where(norms > active_step_threshold)[0]
    if active_indices.size <= 1:
        return {"reverse_ratio": 0.0, "hard_turn_ratio": 0.0}

    cos_values: list[float] = []
    for index in range(active_indices.size - 1):
        first_step = steps[active_indices[index]]
        second_step = steps[active_indices[index + 1]]
        cosine = float(
            np.dot(first_step, second_step)
            / (np.linalg.norm(first_step) * np.linalg.norm(second_step))
        )
        cos_values.append(float(np.clip(cosine, -1.0, 1.0)))

    cos_array = np.asarray(cos_values, dtype=np.float64)
    return {
        "reverse_ratio": float(np.mean(cos_array < 0.0)),
        "hard_turn_ratio": float(np.mean(cos_array < 0.5)),
    }


def gripper_change_metrics(gripper_signal: np.ndarray) -> dict[str, int]:
    """评估夹爪是否存在来回切换。

    `switch_count` 统计总切换次数；
    `chatter_count` 统计短时间内反复来回切换的次数。
    """

    values = np.asarray(gripper_signal, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return {"switch_count": 0, "chatter_count": 0}

    binary = (values >= 0.5).astype(np.int32)
    switch_indices = np.where(np.abs(np.diff(binary)) > 0)[0]
    switch_count = int(switch_indices.size)

    chatter_count = 0
    for index in range(max(0, switch_indices.size - 1)):
        if int(switch_indices[index + 1] - switch_indices[index]) <= 6:
            chatter_count += 1

    return {"switch_count": switch_count, "chatter_count": chatter_count}


def choose_label(score: float) -> str:
    """把数值分数转换成更直观的等级。"""

    if score >= 90.0:
        return "Excellent"
    if score >= 80.0:
        return "Good"
    if score >= 65.0:
        return "Warning"
    return "Severe Jitter"


def build_suggestions(metrics: dict[str, float | int]) -> list[str]:
    """根据指标给出简短诊断建议。"""

    suggestions: list[str] = []

    if float(metrics["ee_jerk_p95"]) > 1.5:
        suggestions.append("End-effector jerk is high; rollout or training may show visible shaking.")
    if float(metrics["ee_reverse_ratio"]) > 0.05:
        suggestions.append("The end-effector path contains frequent reversals; the policy may be over-correcting.")
    if float(metrics["action_jerk_p95"]) > 40.0:
        suggestions.append("Action labels switch too sharply; smooth or interpolate target poses before training.")
    if int(metrics["gripper_chatter_count"]) > 0:
        suggestions.append("The gripper toggles repeatedly in a short window, which can teach grasp hesitation.")
    if float(metrics["action_hold_ratio"]) < 0.05:
        suggestions.append("The action changes almost every frame and lacks stable hold segments, which can amplify jitter.")
    if not suggestions:
        suggestions.append("The trajectory looks stable overall and is suitable as a training baseline.")

    return suggestions


def compute_smoothness_score(metrics: dict[str, float | int]) -> float:
    """根据多个抖动迹象给出一个 0~100 的平滑性分数。"""

    score = 100.0

    ee_step_p95_mm = float(metrics["ee_step_p95_mm"])
    ee_accel_p95 = float(metrics["ee_accel_p95"])
    ee_jerk_p95 = float(metrics["ee_jerk_p95"])
    ee_reverse_ratio = float(metrics["ee_reverse_ratio"])
    ee_hard_turn_ratio = float(metrics["ee_hard_turn_ratio"])
    action_jerk_p95 = float(metrics["action_jerk_p95"])
    switch_count = int(metrics["gripper_switch_count"])
    chatter_count = int(metrics["gripper_chatter_count"])

    score -= min(18.0, max(0.0, ee_step_p95_mm - 8.0) * 1.0)
    score -= min(18.0, max(0.0, ee_accel_p95 - 0.20) * 30.0)
    score -= min(24.0, max(0.0, ee_jerk_p95 - 1.50) * 6.0)
    score -= min(10.0, max(0.0, action_jerk_p95 - 40.0) * 0.25)
    score -= min(15.0, ee_reverse_ratio * 120.0)
    score -= min(8.0, ee_hard_turn_ratio * 60.0)
    score -= min(12.0, chatter_count * 6.0 + max(0, switch_count - 2) * 2.0)

    return float(np.clip(score, 0.0, 100.0))


def evaluate_episode(path: Path, dt: float) -> dict[str, object]:
    """评估单个 episode。"""

    data = np.load(path, allow_pickle=True)
    actions = np.asarray(data["action"], dtype=np.float64)
    action_positions = actions[:, :3]
    action_gripper = actions[:, 3]

    state = np.asarray(data["observation.state"], dtype=np.float64)
    success = bool(np.asarray(data["success"]).item())

    # 17_demo 当前状态格式：
    # 7 关节 + 3 末端位置 + 4 四元数 + 1 夹爪宽度
    ee_positions = state[:, 7:10] if state.shape[1] >= 10 else action_positions.copy()
    gripper_width = state[:, 14] if state.shape[1] >= 15 else np.zeros(state.shape[0], dtype=np.float64)

    action_steps = step_norms(action_positions)
    action_velocities = finite_difference_norms(action_positions, order=1, dt=dt)
    action_accelerations = finite_difference_norms(action_positions, order=2, dt=dt)
    action_jerks = finite_difference_norms(action_positions, order=3, dt=dt)

    ee_steps = step_norms(ee_positions)
    ee_velocities = finite_difference_norms(ee_positions, order=1, dt=dt)
    ee_accelerations = finite_difference_norms(ee_positions, order=2, dt=dt)
    ee_jerks = finite_difference_norms(ee_positions, order=3, dt=dt)

    action_turns = direction_change_metrics(action_positions)
    ee_turns = direction_change_metrics(ee_positions)
    gripper_action_metrics = gripper_change_metrics(action_gripper)
    gripper_width_metrics = gripper_change_metrics(gripper_width)

    metrics: dict[str, float | int] = {
        "frames": int(actions.shape[0]),
        "success": int(success),
        "action_hold_ratio": float(np.mean(action_steps < 1e-6)) if action_steps.size > 0 else 1.0,
        "action_step_mean_mm": summarize_values(action_steps)["mean"] * 1000.0,
        "action_step_p95_mm": summarize_values(action_steps)["p95"] * 1000.0,
        "action_step_max_mm": summarize_values(action_steps)["max"] * 1000.0,
        "action_vel_p95": summarize_values(action_velocities)["p95"],
        "action_accel_p95": summarize_values(action_accelerations)["p95"],
        "action_jerk_p95": summarize_values(action_jerks)["p95"],
        "action_reverse_ratio": action_turns["reverse_ratio"],
        "action_hard_turn_ratio": action_turns["hard_turn_ratio"],
        "ee_step_mean_mm": summarize_values(ee_steps)["mean"] * 1000.0,
        "ee_step_p95_mm": summarize_values(ee_steps)["p95"] * 1000.0,
        "ee_step_max_mm": summarize_values(ee_steps)["max"] * 1000.0,
        "ee_vel_p95": summarize_values(ee_velocities)["p95"],
        "ee_accel_p95": summarize_values(ee_accelerations)["p95"],
        "ee_jerk_p95": summarize_values(ee_jerks)["p95"],
        "ee_reverse_ratio": ee_turns["reverse_ratio"],
        "ee_hard_turn_ratio": ee_turns["hard_turn_ratio"],
        "gripper_switch_count": gripper_action_metrics["switch_count"],
        "gripper_chatter_count": gripper_action_metrics["chatter_count"],
        "gripper_width_switch_count": gripper_width_metrics["switch_count"],
    }

    score = compute_smoothness_score(metrics)
    label = choose_label(score)
    suggestions = build_suggestions(metrics)

    return {
        "episode_file": path.name,
        "smoothness_score": round(score, 3),
        "smoothness_label": label,
        "suggestions": suggestions,
        **metrics,
    }


def plot_episode_smoothness(
    episode_path: Path,
    row: dict[str, object],
    output_dir: Path,
    dt: float,
) -> Path:
    """为单个 episode 画一张动作平滑性总览图。"""

    data = np.load(episode_path, allow_pickle=True)
    actions = np.asarray(data["action"], dtype=np.float64)
    action_positions = actions[:, :3]
    action_gripper = actions[:, 3]

    state = np.asarray(data["observation.state"], dtype=np.float64)
    ee_positions = state[:, 7:10] if state.shape[1] >= 10 else action_positions.copy()
    gripper_width = state[:, 14] if state.shape[1] >= 15 else np.zeros(state.shape[0], dtype=np.float64)

    action_steps_mm = step_norms(action_positions) * 1000.0
    ee_steps_mm = step_norms(ee_positions) * 1000.0
    action_acc = finite_difference_norms(action_positions, order=2, dt=dt)
    ee_acc = finite_difference_norms(ee_positions, order=2, dt=dt)
    action_jerk = finite_difference_norms(action_positions, order=3, dt=dt)
    ee_jerk = finite_difference_norms(ee_positions, order=3, dt=dt)

    frame_t = time_axis(action_positions.shape[0], dt)
    step_t = time_axis(action_steps_mm.shape[0], dt, start_at_one=True)
    acc_t = time_axis(action_acc.shape[0], dt, start_at_one=True)
    jerk_t = time_axis(action_jerk.shape[0], dt, start_at_one=True)

    figure, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.reshape(4, 2)

    axis = axes[0, 0]
    axis.plot(frame_t, action_positions[:, 0], label="target_x", linewidth=1.6)
    axis.plot(frame_t, action_positions[:, 1], label="target_y", linewidth=1.6)
    axis.plot(frame_t, action_positions[:, 2], label="target_z", linewidth=1.6)
    axis.set_title("Target Action Position")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Position / m")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[0, 1]
    axis.plot(frame_t, ee_positions[:, 0], label="ee_x", linewidth=1.6)
    axis.plot(frame_t, ee_positions[:, 1], label="ee_y", linewidth=1.6)
    axis.plot(frame_t, ee_positions[:, 2], label="ee_z", linewidth=1.6)
    axis.set_title("Executed End-Effector Position")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Position / m")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[1, 0]
    axis.plot(step_t, action_steps_mm, label="action_step", linewidth=1.6)
    axis.plot(step_t, ee_steps_mm, label="ee_step", linewidth=1.6)
    axis.axhline(float(row["ee_step_p95_mm"]), color="#cc5500", linestyle="--", linewidth=1.0, label="ee_step_p95")
    axis.set_title("Per-Step Displacement")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Displacement / mm")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[1, 1]
    axis.plot(acc_t, action_acc, label="action_acc", linewidth=1.6)
    axis.plot(acc_t, ee_acc, label="ee_acc", linewidth=1.6)
    axis.axhline(float(row["ee_accel_p95"]), color="#cc5500", linestyle="--", linewidth=1.0, label="ee_acc_p95")
    axis.set_title("Acceleration Norm")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Acceleration")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[2, 0]
    axis.plot(jerk_t, action_jerk, label="action_jerk", linewidth=1.6)
    axis.plot(jerk_t, ee_jerk, label="ee_jerk", linewidth=1.6)
    axis.axhline(float(row["ee_jerk_p95"]), color="#cc5500", linestyle="--", linewidth=1.0, label="ee_jerk_p95")
    axis.set_title("Jerk Norm")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("jerk")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[2, 1]
    axis.step(frame_t, action_gripper, where="post", label="target_gripper_closed", linewidth=1.6)
    axis.plot(frame_t, gripper_width, label="gripper_width", linewidth=1.6)
    axis.set_title("Gripper Target and Width")
    axis.set_xlabel("Time / s")
    axis.set_ylabel("Signal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)

    axis = axes[3, 0]
    axis.plot(action_positions[:, 0], action_positions[:, 1], label="target_xy", linewidth=2.0)
    axis.plot(ee_positions[:, 0], ee_positions[:, 1], label="ee_xy", linewidth=2.0)
    axis.scatter(action_positions[0, 0], action_positions[0, 1], c="green", s=50, label="start")
    axis.scatter(action_positions[-1, 0], action_positions[-1, 1], c="red", s=50, label="end")
    axis.set_title("Top-Down Path")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    axis.axis("equal")

    axis = axes[3, 1]
    axis.axis("off")
    text_lines = [
        f"episode: {row['episode_file']}",
        f"score: {row['smoothness_score']}  ({row['smoothness_label']})",
        f"success: {bool(int(row['success']))}",
        "",
        f"action_step_p95: {float(row['action_step_p95_mm']):.2f} mm",
        f"action_jerk_p95: {float(row['action_jerk_p95']):.3f}",
        f"ee_step_p95: {float(row['ee_step_p95_mm']):.2f} mm",
        f"ee_accel_p95: {float(row['ee_accel_p95']):.3f}",
        f"ee_jerk_p95: {float(row['ee_jerk_p95']):.3f}",
        f"ee_reverse_ratio: {float(row['ee_reverse_ratio']):.3f}",
        f"gripper_switch: {int(row['gripper_switch_count'])}",
        f"gripper_chatter: {int(row['gripper_chatter_count'])}",
        "",
        "notes:",
        *[f"- {suggestion}" for suggestion in row["suggestions"]],
    ]
    axis.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    figure.suptitle(
        f"Franka Action Smoothness Overview: {row['episode_file']}",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{episode_path.stem}_smoothness.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return plot_path


def plot_score_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """绘制所有 episode 的总分对比图。"""

    sorted_rows = sorted(rows, key=lambda row: float(row["smoothness_score"]))
    labels = [str(row["episode_file"]).replace(".npz", "") for row in sorted_rows]
    scores = [float(row["smoothness_score"]) for row in sorted_rows]

    figure, axis = plt.subplots(figsize=(max(10, len(rows) * 1.4), 5))
    colors = [
        "#2a9d8f" if score >= 90.0 else "#e9c46a" if score >= 80.0 else "#f4a261" if score >= 65.0 else "#e76f51"
        for score in scores
    ]
    axis.bar(labels, scores, color=colors)
    axis.set_ylim(0.0, 100.0)
    axis.set_title("Episode Smoothness Scores")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Score")
    axis.grid(True, axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "smoothness_scores.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return plot_path


def resolve_episode_paths(input_dir: Path, episode_index: int | None) -> list[Path]:
    """解析待评估的 episode 路径列表。"""

    if episode_index is not None:
        path = input_dir / f"episode_{episode_index:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"找不到指定 episode: {path}")
        return [path]

    paths = sorted(input_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"目录里没有找到 episode 文件: {input_dir}")
    return paths


def write_csv_report(path: Path, rows: list[dict[str, object]]) -> None:
    """写出 CSV 报表。"""

    if not rows:
        return

    fieldnames = [
        "episode_file",
        "smoothness_score",
        "smoothness_label",
        "frames",
        "success",
        "action_hold_ratio",
        "action_step_p95_mm",
        "action_accel_p95",
        "action_jerk_p95",
        "action_reverse_ratio",
        "ee_step_p95_mm",
        "ee_accel_p95",
        "ee_jerk_p95",
        "ee_reverse_ratio",
        "ee_hard_turn_ratio",
        "gripper_switch_count",
        "gripper_chatter_count",
        "gripper_width_switch_count",
        "suggestions",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {fieldname: row.get(fieldname, "") for fieldname in fieldnames}
            csv_row["suggestions"] = " | ".join(row["suggestions"])
            writer.writerow(csv_row)


def print_terminal_summary(rows: list[dict[str, object]]) -> None:
    """在终端打印摘要。"""

    scores = np.asarray([float(row["smoothness_score"]) for row in rows], dtype=np.float64)
    sorted_rows = sorted(rows, key=lambda row: float(row["smoothness_score"]))

    print(f"共评估 {len(rows)} 条 episode")
    print(
        "分数统计:"
        f" mean={np.mean(scores):.2f}"
        f" p50={np.percentile(scores, 50):.2f}"
        f" p95={np.percentile(scores, 95):.2f}"
        f" min={np.min(scores):.2f}"
        f" max={np.max(scores):.2f}"
    )
    print("最需要优先检查的 3 条数据:")

    for row in sorted_rows[:3]:
        print(
            f"  {row['episode_file']}"
            f" score={row['smoothness_score']}"
            f" label={row['smoothness_label']}"
            f" ee_jerk_p95={float(row['ee_jerk_p95']):.3f}"
            f" action_jerk_p95={float(row['action_jerk_p95']):.3f}"
            f" gripper_chatter={int(row['gripper_chatter_count'])}"
        )
        for suggestion in row["suggestions"]:
            print(f"    - {suggestion}")


def main() -> None:
    """脚本主入口。"""

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 某些受限环境里，仓库目录可能是只读的。
        # 这时自动回退到 /tmp，避免评估逻辑本身被文件系统权限卡住。
        output_dir = FALLBACK_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录不可写，已回退到: {output_dir}")

    episode_paths = resolve_episode_paths(input_dir, args.episode)
    rows = [evaluate_episode(path, dt=DEFAULT_DT) for path in episode_paths]
    rows = sorted(rows, key=lambda row: str(row["episode_file"]))

    csv_path = output_dir / "smoothness_report.csv"
    json_path = output_dir / "smoothness_report.json"

    plot_paths = [plot_episode_smoothness(path, row, output_dir, DEFAULT_DT) for path, row in zip(episode_paths, rows)]
    summary_plot_path = plot_score_summary(rows, output_dir)

    write_csv_report(csv_path, rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_terminal_summary(rows)
    print(f"CSV 报表: {csv_path}")
    print(f"JSON 报表: {json_path}")
    print(f"总分图: {summary_plot_path}")
    if plot_paths:
        print(f"单条轨迹图示例: {plot_paths[0]}")


if __name__ == "__main__":
    main()

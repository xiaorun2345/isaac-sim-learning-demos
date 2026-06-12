"""在独立 Python 3.12 环境中加载 SmolVLA，并通过本地 socket 给 Isaac 提供动作推理。"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import traceback
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True, help="SmolVLA policy 目录。")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址。")
    parser.add_argument("--port", type=int, default=5567, help="服务监听端口。")
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/third_party/lerobot/src"),
        help="本地 LeRobot 源码目录。",
    )
    parser.add_argument(
        "--fallback-hf-cache",
        type=Path,
        default=Path("/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/.cache/huggingface"),
        help="已有 Hugging Face 缓存目录。",
    )
    return parser.parse_args()


ARGS = parse_args()
SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_SERVER_PROTOCOL_VERSION = "smolvla_socket_v2"


def prepare_offline_hf_cache(script_dir: Path) -> None:
    local_cache_dir = script_dir / ".cache" / "huggingface"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(local_cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(local_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(local_cache_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    local_vlm_cache = local_cache_dir / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    fallback_vlm_cache = ARGS.fallback_hf_cache / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    if not local_vlm_cache.exists() and fallback_vlm_cache.is_dir():
        local_vlm_cache.symlink_to(fallback_vlm_cache)


prepare_offline_hf_cache(SCRIPT_DIR)

if ARGS.lerobot_src.is_dir():
    import sys

    sys.path.insert(0, str(ARGS.lerobot_src))

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, policy_action_to_transition, transition_to_batch, transition_to_policy_action


def resolve_policy_dir(policy_dir: Path) -> Path:
    resolved = policy_dir.resolve()
    if (resolved / "config.json").is_file() and (resolved / "model.safetensors").is_file():
        return resolved
    nested = resolved / "pretrained_model"
    if (nested / "config.json").is_file() and (nested / "model.safetensors").is_file():
        return nested
    raise FileNotFoundError(f"Cannot find policy files under: {policy_dir}")


def load_policy_bundle(policy_dir: Path):
    policy_dir = resolve_policy_dir(policy_dir)
    config = PreTrainedConfig.from_pretrained(str(policy_dir), local_files_only=True)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.device != "cuda":
        config.use_amp = False
    if config.type != "smolvla":
        raise ValueError(f"Only smolvla policy is supported, got: {config.type}")

    policy = SmolVLAPolicy.from_pretrained(
        str(policy_dir),
        config=config,
        local_files_only=True,
    )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir),
        config_filename="policy_preprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": config.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(policy_dir),
        config_filename="policy_postprocessor.json",
        local_files_only=True,
        overrides={"device_processor": {"device": config.device}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    print(f"[policy-server] loaded policy: {policy_dir}", flush=True)
    print(f"[policy-server] policy type: {config.type}", flush=True)
    print(f"[policy-server] policy device: {config.device}", flush=True)
    return policy, preprocessor, postprocessor, config.device, policy_dir


def predict_policy_action(
    observation: dict,
    task: str,
    robot_type: str,
    policy,
    preprocessor,
    postprocessor,
    device: str,
):
    observation = prepare_observation_for_inference(
        observation=observation,
        device=torch.device(device),
        task=task,
        robot_type=robot_type,
    )
    observation = preprocessor(observation)
    with torch.inference_mode():
        action = policy.select_action(observation)
    action = postprocessor(action)
    return action.squeeze(0).to("cpu").numpy().astype("float32")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Client socket closed unexpectedly.")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_pickle_message(sock: socket.socket) -> dict:
    body_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    return pickle.loads(recv_exact(sock, body_size))


def send_pickle_message(sock: socket.socket, payload: dict) -> None:
    body = pickle.dumps(payload, protocol=4)
    sock.sendall(struct.pack("!I", len(body)))
    sock.sendall(body)


def handle_request(request: dict, policy, preprocessor, postprocessor, device: str, policy_dir: Path) -> dict:
    request_type = request.get("type")
    if request_type == "ping":
        return {
            "status": "ok",
            "policy_dir": str(policy_dir),
            "device": device,
            "pid": os.getpid(),
            "protocol_version": POLICY_SERVER_PROTOCOL_VERSION,
        }
    if request_type == "reset":
        policy.reset()
        return {"status": "ok"}
    if request_type == "predict":
        action = predict_policy_action(
            observation=request["observation"],
            task=request["task"],
            robot_type=request["robot_type"],
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
        )
        return {"status": "ok", "action": action}
    raise ValueError(f"Unsupported request type: {request_type}")


def main() -> None:
    policy, preprocessor, postprocessor, device, policy_dir = load_policy_bundle(ARGS.policy_dir)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((ARGS.host, ARGS.port))
        server.listen(8)
        print(f"[policy-server] listening on {ARGS.host}:{ARGS.port}", flush=True)

        while True:
            conn, addr = server.accept()
            with conn:
                request = {}
                try:
                    request = recv_pickle_message(conn)
                    response = handle_request(request, policy, preprocessor, postprocessor, device, policy_dir)
                except Exception as exc:
                    response = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    }
                send_pickle_message(conn, response)
                if request.get("type") == "predict":
                    print(f"[policy-server] predict served for {addr[0]}:{addr[1]}", flush=True)


if __name__ == "__main__":
    main()

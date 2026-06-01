# 13 GR00T Inference

This demo makes the GR00T inference stage explicit:

1. build normalized VLA observation
2. load checkpoint path
3. create policy wrapper
4. run inference
5. decode the action semantics

Set the checkpoint path before running:

```powershell
$env:GROOT_CHECKPOINT="C:\\path\\to\\checkpoint"
```

NVIDIA references for the real deployment pattern:

- [Post-training With GR00T](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-for-healthcare/latest/training-healthcare-robots-from-scratch/04-model-flywheel/02-post-training-with-gr00t.html)
- [Rollout a VLA Model in Simulation](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-for-healthcare/latest/training-healthcare-robots-from-scratch/04-model-flywheel/03-deploy.html)

There are two scripts:

- `demo.py`: offline teaching mode with a deterministic stand-in policy
- `policy_client_demo.py`: real GR00T `PolicyClient` using the official server-client API

Start the real GR00T N1.7 policy server from an Isaac-GR00T checkout:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Then run the client:

```powershell
$env:GROOT_HOST="127.0.0.1"
$env:GROOT_PORT="5555"
python policy_client_demo.py
```

The observation dictionary must match the selected embodiment configuration. The sample keys are intentionally easy to locate and adapt:

- `video.wrist_cam`
- `state.joints`
- `language.task`

Official GR00T reference:

- [Understanding the GR00T Policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md)

# Isaac Sim 15 Demo Design

## Goals

Create a progressive series of 15 standalone demos for learning NVIDIA Isaac Sim, ROS 2 integration, Isaac Lab, and GR00T-based VLA inference. The demos should start from a minimal standalone Python scene and build toward an Isaac Lab example with GR00T-style language-conditioned operation.

## Audience

This series is aimed at learners who want hands-on, runnable examples rather than only conceptual descriptions. The examples should stay readable, concise, and easy to extend.

## Scope

This series covers:

- Isaac Sim standalone Python application bootstrapping
- creating scenes with ground, lighting, and primitive objects
- basic rigid-body simulation and scene composition
- camera creation and image capture
- Franka robot loading and low-level joint control
- gripper control and scripted motion sequences
- cube grasping and simple state machine logic
- Isaac Sim ROS 2 communication
- transition into Isaac Lab
- Isaac Lab observation pipeline
- GR00T-oriented VLA inference demos in Isaac Lab

This series does not attempt:

- training a VLA model from scratch
- guaranteeing local availability of specific external model checkpoints
- hiding version differences between Isaac Sim, Isaac Lab, ROS 2, and GR00T releases

## Structure

The project should use a flat, predictable teaching layout:

- `README.md`
  - top-level learning map, prerequisites, and run instructions
- `common/`
  - shared helper modules used across multiple demos
- `01_...` through `15_...`
  - one folder per demo
  - each folder contains `README.md` and `demo.py`
- `docs/superpowers/specs/`
  - design documents
- `docs/superpowers/plans/`
  - implementation plans

## Demo order

### 01. Minimal Isaac Sim standalone scene

Purpose:
Launch the simulator from Python and run an empty or near-empty scene with a ground plane.

Key concepts:

- `SimulationApp`
- `World`
- stepping the simulation
- graceful shutdown

### 02. Add lights and primitive objects

Purpose:
Create a visible scene with a light source and a few basic objects.

Key concepts:

- stage authoring
- primitive geometry
- transforms
- visual debugging

### 03. Multi-object scene with physics properties

Purpose:
Add several dynamic and static objects with simple rigid-body properties.

Key concepts:

- rigid bodies
- colliders
- mass and basic physics behavior

### 04. Add a camera and read images

Purpose:
Place a camera in the scene and retrieve rendered frames in Python.

Key concepts:

- camera prims
- render products or sensor interfaces
- image array access

### 05. Load Franka and control joints

Purpose:
Load Franka Panda and issue direct articulation commands.

Key concepts:

- `SingleArticulation`
- `ArticulationAction`
- joint position control
- joint indexing

### 06. Control the gripper and run an action sequence

Purpose:
Execute a simple time-based or step-based motion script for arm and gripper.

Key concepts:

- composing actions
- opening and closing the gripper
- simple sequencing logic

### 07. Pick a cube with Franka

Purpose:
Add a cube and perform a scripted pick motion.

Key concepts:

- task geometry setup
- pre-grasp, grasp, lift stages
- simple success criteria

### 08. Pick and place with a state machine

Purpose:
Refactor scripted motion into a clearer pick-and-place state machine.

Key concepts:

- state transitions
- task logic separation
- reset behavior

### 09. Isaac Sim ROS 2 basics

Purpose:
Publish simulation state and receive a minimal ROS 2 message path.

Key concepts:

- ROS 2 bridge prerequisites
- publishers/subscribers
- timing and synchronization

### 10. ROS 2 control for Franka

Purpose:
Use ROS 2 communication to send or emulate robot commands in Isaac Sim.

Key concepts:

- command topics
- robot state topics
- simple external control loop

### 11. Isaac Lab minimal manipulation scene

Purpose:
Introduce Isaac Lab structure while preserving a familiar manipulation setup.

Key concepts:

- Isaac Lab app launcher pattern
- environment/config structure
- asset spawning

### 12. Build a VLA-ready observation pipeline

Purpose:
Prepare camera images, text instructions, and robot state into a VLA-friendly input package.

Key concepts:

- observation dictionaries
- prompt strings
- tensor or numpy packaging

### 13. Connect GR00T inference

Purpose:
Run a GR00T-oriented inference path that consumes image plus instruction and emits an action representation.

Key concepts:

- policy wrapper abstraction
- model input normalization
- action decoding

### 14. GR00T-driven pick and place in Isaac Lab

Purpose:
Use model output to drive a simple manipulation loop in Isaac Lab.

Key concepts:

- closed-loop inference
- action application
- loop timing

### 15. Multi-instruction VLA task demo

Purpose:
Show a richer language-conditioned manipulation example, such as selecting an object by color and placing it into a target zone.

Key concepts:

- instruction variation
- reusable task scaffolding
- evaluation logging

## Architecture

The codebase should emphasize teaching clarity over abstraction depth. Each demo must run independently, but shared utilities should be extracted when duplication becomes distracting rather than educational.

Three layers are enough:

1. Demo scripts
Each `demo.py` should be the main teaching artifact and keep the control flow obvious.

2. Shared helpers
The `common/` package should hold lightweight utilities, such as reusable Franka setup helpers, camera helpers, simple state machines, ROS 2 message wrappers, and GR00T policy adapters.

3. Documentation
Each demo README should explain what the demo adds compared with the previous one, how to run it, and what to look for.

## VLA / GR00T Strategy

The GR00T portion should prefer real inference-oriented structure rather than a purely fake placeholder pipeline, because the goal is to learn how a real VLA stack is wired into simulation.

However, the implementation should still be robust to missing checkpoints or unavailable local dependencies:

- the wrapper should isolate model-specific imports
- the docs should clearly note required checkpoints and environment setup
- the code should fail with actionable messages if GR00T dependencies are not installed
- where practical, the demos should support a dry-run or stub mode so the rest of the Isaac Lab pipeline remains understandable

This keeps the demos educational even if a learner has not yet downloaded the full model assets.

## Error Handling

The demos should fail early and clearly:

- missing Isaac Sim or Isaac Lab imports should print setup guidance
- missing ROS 2 bridge requirements should point to the relevant demo README
- missing GR00T checkpoints should explain what path or environment variable is expected

Errors should favor short, direct guidance over complex recovery logic.

## Testing Strategy

This repository is primarily educational and many demos depend on heavyweight local environments. Testing should therefore focus on what can be validated cheaply and reliably:

- Python syntax checks for all created scripts
- import-guard checks for helper modules where possible
- doc consistency checks by manual review

Full end-to-end simulation execution is valuable but should be treated as environment-dependent verification, not assumed as universally runnable inside this workspace.

## Versioning Assumptions

Because NVIDIA robotics tooling evolves quickly, the docs should avoid hard-coding unnecessary version-specific claims. Instead, they should:

- name the expected tools: Isaac Sim, Isaac Lab, ROS 2, GR00T
- explain when code may require minor path or import adjustments
- keep demo logic easy to adapt

## Deliverables

The implementation phase should create:

- one top-level `README.md`
- one `common/` helper package
- fifteen demo directories with runnable scripts and short READMEs
- one implementation plan document

## Success Criteria

The project is successful if:

- a learner can read `README.md` and understand the full sequence
- each demo adds a clear new concept
- `01-10` teach Isaac Sim and ROS 2 fundamentals progressively
- `11-15` teach Isaac Lab plus GR00T-style VLA integration progressively
- the generated files are small enough to study and modify

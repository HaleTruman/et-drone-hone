# Drone Vision + Autonomy Training Framework

This repository provides the **core machine learning and control framework** used to train and operate an autonomous drone within a simulation environment. The project focuses on **visual perception, reinforcement learning flight behavior, and modular control systems** that can interface with external simulation tooling.

The system is designed to support **training in Unreal Engine environments**, allowing the drone to perceive objects (such as navigation rings) through computer vision and learn optimal flight behaviors through reinforcement learning.

---

## Overview

The repository contains the primary infrastructure for:

- **Visual perception modeling**
- **Reinforcement learning flight policies**
- **Drone control abstraction**
- **Path optimization and decision layers**
- **Training pipelines for simulation environments**

The architecture separates **perception, decision-making, and control**, enabling each component to be trained and improved independently.

---

## System Architecture

The drone autonomy system is composed of several layers:

### 1. Perception (CNN)

A convolutional neural network is trained to interpret visual input from the simulation.

Responsibilities include:

- Detecting navigation targets (rings)
- Identifying environmental obstacles
- Estimating spatial orientation and distance
- Producing structured scene representations

Training data is generated from Unreal Engine environments with associated metadata labels.

---

### 2. Decision Policy (Reinforcement Learning)

A reinforcement learning model determines how the drone should move through the environment.

The policy learns to:

- Navigate through target rings
- Optimize trajectory and speed
- Avoid obstacles
- Maximize reward signals based on flight performance

The RL system operates on outputs from the perception layer.

---

### 3. Drone Control Model

A lower-level control module translates high-level motion commands into physical drone actions.

This includes:

- Propeller thrust modeling
- Torque and rotational control
- Stabilization behaviors
- Motion translation into motor commands

The control layer abstracts the underlying drone physics so the higher-level policy can operate in simplified control space.

---

### 4. Path Optimization

Additional modeling supports trajectory optimization, including:

- Ideal spline path calculation
- Flight efficiency metrics
- Reward shaping for RL training

This layer helps guide learning toward efficient flight paths.

---

## Unreal Engine Integration

Simulation control is performed through **external tooling** that connects this repository to Unreal Engine.

Key components outside this repo include:

- **WebSocket bridge**
- **Python control interface**
- **Unreal Engine simulation environment**

The bridge allows:

- Real-time drone control
- Sensor and telemetry streaming
- Image capture for CNN inference
- Reinforcement learning training loops

---

## Training Pipelines

The repository supports two primary training workflows.

### CNN Training

- Image dataset generation from Unreal Engine
- Metadata annotation (JSON labels)
- Object detection training
- Spatial prediction modeling

### Reinforcement Learning Training

- Drone flight simulation
- Reward-based policy learning
- Path optimization feedback
- Iterative policy improvement

---

## Design Goals

This system is built around several core principles:

- **Modularity** – perception, policy, and control are separate layers
- **Simulation-first training** – all models are developed in Unreal environments
- **Hardware abstraction** – control models can adapt to different drone platforms
- **Real-time performance** – inference designed for onboard compute
- **Extensibility** – architecture supports future autonomy features

---

## Repository Scope

This repository focuses on **machine learning models, training pipelines, and drone control abstractions**.

It does **not** contain:

- Unreal Engine simulation scenes
- WebSocket bridge implementation
- External control server infrastructure

Those components live in separate tooling repositories.

---

## Future Work

Planned extensions include:

- Online model adaptation
- Improved sim-to-real transfer
- Dynamic environment generalization
- Multi-drone coordination
- Advanced perception models

---

## Summary

This repository serves as the **core autonomy framework** for a simulated drone system capable of:

- Perceiving environments through computer vision
- Learning flight behavior through reinforcement learning
- Executing optimized flight paths through modular control systems

It forms the central ML and control layer that interfaces with Unreal Engine simulations and external runtime tooling.
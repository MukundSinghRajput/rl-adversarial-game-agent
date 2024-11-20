# Reinforcement Learning Adversarial Game Agent

A lightweight, hybrid artificial intelligence web application that allows users to play Tic-Tac-Toe against two distinct machine learning paradigms: **Q-Learning (Reinforcement Learning)** and **Minimax with Alpha-Beta Pruning**.

The backend is built using Python and Flask, utilizing a native Q-learning implementation that trains entirely via self-play without relying on external heavy machine learning frameworks like TensorFlow or PyTorch.

---

## Features

* **Hybrid AI Architecture**:
* **Hard Difficulty**: Utilizes a deterministic Minimax algorithm optimized with Alpha-Beta pruning, ensuring mathematically flawless execution.
* **Medium & Easy Difficulties**: Implements a probabilistic Q-Learning agent using variable $\epsilon$-greedy exploration rates to mimic human-like errors.


* **Autonomous Self-Play Training**: Automatically executes a $200,000$ episode self-play routine on the initial launch to populate the Q-table matrix if a saved model is not detected.
* **Persistent State**: Serializes and caches the trained policy locally using Python's `pickle` protocol for near-instantaneous startup on subsequent runs.
* **Automated Environment Provisioning**: Embedded system-level routing handles local port instantiation and invokes default web browsers automatically upon server binding.

---

## Technical Stack Overview

### 1. Reinforcement Learning (Q-Learning)

The agent updates its action-value function matrix, $Q(s, a)$, based on the Bellman Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:

* Learning Rate ($\alpha$) = `0.3`
* Discount Factor ($\gamma$) = `0.9`
* Exploration vs. Exploitation Strategy: $\epsilon$-decay over $200,000$ iterations down to a floor value of `0.05`.

### 2. Adversarial Search (Minimax)

To guarantee unbeatable play on the highest difficulty setting, the engine drops into a recursive depth-penalized search space evaluated between $\alpha = -\infty$ and $\beta = \infty$. The dynamic score function rewards quicker victories and penalizes delayed losses:

* AI Win: $10 - \text{depth}$
* Human Win: $\text{depth} - 10$

---

## Installation & Environment Setup

### Prerequisites

* Python 3.8 or higher
* Pip package manager or you can use UV which i am also using.

### 1. Clone the Repository

```bash
git clone https://github.com/MukundSinghRajput/rl-adversarial-game-agent
cd rl-adversarial-game-agent

```

### 2. Install Dependencies

This project uses standard library modules for the core AI components and Flask for web routing. Install the dependencies via pip:

```bash
pip install Flask

```

### 3. Missing Structural Components

Before launching the server, ensure your directory layout includes the HTML template file expected by the Flask application layer:

```text
├── main.py
└── templates/
    └── index.html
```

---

## Running the Application

Execute the entry point script from your terminal:

```bash
python main.py
```

### Initial Initialization behavior

If `q_table.pkl` does not exist in the root directory, the terminal will intercept launch routing to execute the $200,000$ game training phase. Progress indicators will log telemetry results at $50,000$ episode intervals:

```text
🎓  Training AI for 200,000 episodes ... (one-time setup)

   Ep    50,000/200,000 | Win 31.2%  Draw 57.1%  Loss 11.7%  (ε=0.688)
   Ep   100,000/200,000 | Win 38.4%  Draw 54.0%  Loss  7.6%  (ε=0.375)
   ...
✅  Training complete — model saved to 'q_table.pkl'

```

Once tracking operations finish or when loading an existing cache file, the server exposes an API layer over `[http://127.0.0.1:5000](http://127.0.0.1:5000)` and dispatches an interrupt flag to prompt your computer's OS to launch the default web browser interface automatically.

---

## API Documentation

The server exposes a stateless communication interface processing structural JSON strings.

### Move Request Pipeline

* **Endpoint**: `/ai_move`
* **Method**: `POST`
* **Content-Type**: `application/json`

#### Request Payload Body Example

The board payload expects a flat 1D array structure of length 9, where `0` represents an empty cell, `1` represents the human player token, and `2` represents the AI player token.

```json
{
  "board": [1, 0, 0, 0, 2, 0, 1, 0, 0],
  "difficulty": "medium"
}
```

#### Response Payload Body Example

Returns the selected absolute 0-indexed index value target layout matching optimal constraints.

```json
{
  "move": 2
}
```
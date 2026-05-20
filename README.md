# ZeroMind AI System

ZeroMind is a web-based Gomoku and Checkers AI platform. It combines a Flask
and Socket.IO backend, a Vue-based browser interface, PyTorch policy-value
networks, AlphaZero-style Monte Carlo Tree Search, and tactical guidance for
gameplay and training.

## Features

- Gomoku 15x15 and Checkers 8x8 gameplay
- Player-versus-AI and online player-versus-player modes
- AI difficulty settings
- User registration and login
- Friend list, friend requests, chat, and game challenges
- Self-play training for Gomoku and Checkers
- Training continuation with model, buffer, and state files
- Optional CUDA/GPU support for PyTorch training

## Requirements

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

The web application also requires MongoDB. Configure it with environment
variables or copy `.env.example` as a reference.

## Run The Web App

```powershell
$env:SERVER_PORT="9527"
python service.py
```

Open the browser at:

```text
http://localhost:9527
```

## Train Gomoku

```powershell
python train\train.py --profile balanced --game-batches 4000 --playout-schedule 1:80,1700:120,3000:180 --mcts-batch-size 8
```

Continue from the latest saved Gomoku training state:

```powershell
python train\train.py --continue-last --profile balanced --game-batches 4000 --mcts-batch-size 8
```

## Train Checkers

```powershell
python train\checker_train.py --profile balanced --game-batches 4000 --tactic-ratio 0.25
```

Continue from the latest saved Checkers training state:

```powershell
python train\checker_train.py --continue-last --profile balanced --game-batches 4000 --tactic-ratio 0.25
```

## Model Files

Model weights (`*.model`), training buffers (`*.pkl`), and training state files
are intentionally ignored by Git because they are large generated artifacts.
If trained models are needed for a demo, publish them separately through GitHub
Releases or Git LFS.

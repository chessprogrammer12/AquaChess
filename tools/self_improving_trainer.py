#!/usr/bin/env python3
"""
Incremental NNUE trainer for AquaChess self-improvement after games.
Allows the engine to improve its evaluation by learning from played games.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, Tuple

class IncrementalNNUETrainer:
    """Train NNUE weights incrementally from game outcomes."""
    
    def __init__(self, weights_path: Optional[Path] = None, hidden: int = 64):
        self.hidden = hidden
        self.input_dim = 768
        
        if weights_path and weights_path.exists():
            self.load_weights(weights_path)
        else:
            self._init_weights()
        
        self.version = 1
        self.games_trained = 0
        self.metadata = {
            "created": datetime.now().isoformat(),
            "games_trained": 0,
            "total_loss": 0.0,
            "version": self.version
        }
    
    def _init_weights(self):
        """Initialize weights for a new network."""
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0.0, 0.03, size=(self.input_dim, self.hidden)).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden), dtype=np.float32)
        self.W2 = rng.normal(0.0, 0.03, size=(self.hidden, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)
        
        # Momentum terms for SGD
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
    
    def load_weights(self, weights_path: Path):
        """Load weights from AQUA_NNUE_LITE format."""
        with open(weights_path, 'r') as f:
            lines = f.readlines()
        
        header = lines[0].strip().split()
        if header[0] != "AQUA_NNUE_LITE_V1":
            raise ValueError("Invalid weight file format")
        
        b1_i = np.array(list(map(int, lines[1].split())), dtype=np.int32)
        w1_flat = np.array(list(map(int, lines[2].split())), dtype=np.int32)
        b2_i = int(lines[3])
        w2_i = np.array(list(map(int, lines[4].split())), dtype=np.int32)
        
        # Dequantize
        s1, s2 = 64.0, 64.0
        self.W1 = w1_flat.reshape(self.hidden, self.input_dim).T.astype(np.float32) / s1
        self.b1 = b1_i.astype(np.float32).reshape(1, -1) / s1
        self.W2 = w2_i.astype(np.float32).reshape(-1, 1) / s2
        self.b2 = np.array([[b2_i]], dtype=np.float32) / s2
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.version += 1
    
    def save_weights(self, out_path: Path):
        """Save weights in AQUA_NNUE_LITE format."""
        s1, s2 = 64.0, 64.0
        w1_i = np.clip(np.round(self.W1.T * s1), -2048, 2047).astype(np.int32)
        b1_i = np.clip(np.round(self.b1.reshape(-1) * s1), -100000, 100000).astype(np.int32)
        w2_i = np.clip(np.round(self.W2.reshape(-1) * s2), -4096, 4095).astype(np.int32)
        b2_i = int(np.clip(round(self.b2.item() * s2), -1000000, 1000000))
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write("AQUA_NNUE_LITE_V1 768 64\n")
            f.write(" ".join(map(str, b1_i.tolist())) + "\n")
            f.write(" ".join(map(str, w1_i.reshape(-1).tolist())) + "\n")
            f.write(f"{b2_i}\n")
            f.write(" ".join(map(str, w2_i.tolist())) + "\n")
        
        self.metadata["version"] = self.version
        self.metadata["last_updated"] = datetime.now().isoformat()
        self.metadata["games_trained"] = self.games_trained
    
    def train_from_game(self, positions: np.ndarray, outcome: float, lr: float = 0.001, momentum: float = 0.9):
        """
        Train from a single game's positions.
        
        Args:
            positions: [N, 768] array of board features
            outcome: Game result (-1 = loss, 0 = draw, 1 = win) from engine perspective
            lr: Learning rate
            momentum: Momentum coefficient
        """
        if positions.shape[1] != 768:
            raise ValueError(f"Expected 768 features, got {positions.shape[1]}")
        
        for x, y_target in zip(positions, [outcome] * len(positions)):
            x = x.reshape(1, -1)
            
            # Forward pass
            h_pre = x @ self.W1 + self.b1
            h = np.maximum(h_pre, 0.0)
            pred = h @ self.W2 + self.b2
            
            # Loss
            err = pred - y_target
            loss = float(err * err)
            self.metadata["total_loss"] += loss
            
            # Backward pass
            grad_pred = 2.0 * err
            gW2 = h.T @ grad_pred
            gb2 = grad_pred.copy()
            
            gh = grad_pred @ self.W2.T
            gh[h_pre <= 0.0] = 0.0
            gW1 = x.T @ gh
            gb1 = gh.copy()
            
            # Momentum update
            self.v_W2 = momentum * self.v_W2 - lr * gW2
            self.v_b2 = momentum * self.v_b2 - lr * gb2
            self.v_W1 = momentum * self.v_W1 - lr * gW1
            self.v_b1 = momentum * self.v_b1 - lr * gb1
            
            self.W2 += self.v_W2
            self.b2 += self.v_b2
            self.W1 += self.v_W1
            self.b1 += self.v_b1
        
        self.games_trained += 1
    
    def train_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, lr: float = 0.001, momentum: float = 0.9):
        """Train on a batch of positions."""
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            for idx in perm:
                self.train_from_game(X[idx:idx+1], y[idx], lr, momentum)


def main():
    ap = argparse.ArgumentParser(description="Incremental NNUE trainer for AquaChess")
    ap.add_argument("--weights", type=str, default="", help="Path to existing weights")
    ap.add_argument("--train-data", type=str, required=True, help="Path to .npz with X [N,768], y [N]")
    ap.add_argument("--out", type=str, required=True, help="Output weights file")
    ap.add_argument("--epochs", type=int, default=1, help="Training epochs per batch")
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    ap.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient")
    args = ap.parse_args()
    
    trainer = IncrementalNNUETrainer(
        Path(args.weights) if args.weights else None
    )
    
    data = np.load(args.train_data)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    
    print(f"Training on {len(X)} positions...")
    trainer.train_batch(X, y, epochs=args.epochs, lr=args.lr, momentum=args.momentum)
    
    out_path = Path(args.out)
    trainer.save_weights(out_path)
    print(f"Saved weights to {out_path}")
    print(f"Games trained: {trainer.games_trained}")
    print(f"Total loss: {trainer.metadata['total_loss']:.6f}")


if __name__ == "__main__":
    main()
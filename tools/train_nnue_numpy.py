#!/usr/bin/env python3
"""
Train AquaChess NNUE-lite weights with NumPy.

Outputs text format expected by engine:
AQUA_NNUE_LITE_V1 768 64
<64 hidden biases>
<64*768 input->hidden integer weights>
<1 output bias>
<64 hidden->output integer weights>

Two input modes:
1) --npz data.npz (expects arrays X [N,768], y [N])
2) --pgn games.pgn (requires python-chess; labels from game result)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def piece_index(piece_type: int, color_is_white: bool) -> int:
    # engine order: P,N,B,R,Q,K,p,n,b,r,q,k
    base = 0 if color_is_white else 6
    return base + (piece_type - 1)


def board_to_feature(board) -> np.ndarray:
    # python-chess square indexing matches a1=0..h8=63
    x = np.zeros(768, dtype=np.float32)
    for sq, p in board.piece_map().items():
        idx = piece_index(p.piece_type, p.color) * 64 + sq
        x[idx] = 1.0
    return x


def load_from_pgn(path: Path, max_positions: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import chess.pgn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("python-chess is required for --pgn mode") from exc

    xs = []
    ys = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result == "1-0":
                y_game = 1.0
            elif result == "0-1":
                y_game = -1.0
            elif result == "1/2-1/2":
                y_game = 0.0
            else:
                continue

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply % stride != 0:
                    continue
                xs.append(board_to_feature(board))
                ys.append(y_game)
                count += 1
                if count >= max_positions:
                    break
            if count >= max_positions:
                break

    if not xs:
        raise RuntimeError("No training positions loaded from PGN")

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y


def load_data(args) -> tuple[np.ndarray, np.ndarray]:
    if args.npz:
        data = np.load(args.npz)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        if X.ndim != 2 or X.shape[1] != 768:
            raise ValueError("X must have shape [N,768]")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must have shape [N]")
        return X, y
    if args.pgn:
        return load_from_pgn(Path(args.pgn), args.max_positions, args.stride)
    raise ValueError("Provide either --npz or --pgn")


def train(X: np.ndarray, y: np.ndarray, hidden: int, epochs: int, lr: float, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # targets are in [-1,1] (for pgn mode) or arbitrary (npz mode)
    y = y.reshape(-1, 1)

    W1 = rng.normal(0.0, 0.03, size=(d, hidden)).astype(np.float32)
    b1 = np.zeros((1, hidden), dtype=np.float32)
    W2 = rng.normal(0.0, 0.03, size=(hidden, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        Xs = X[perm]
        ys = y[perm]
        total_loss = 0.0

        for i in range(0, n, batch_size):
            xb = Xs[i : i + batch_size]
            yb = ys[i : i + batch_size]

            h_pre = xb @ W1 + b1
            h = np.maximum(h_pre, 0.0)
            pred = h @ W2 + b2

            err = pred - yb
            loss = float(np.mean(err * err))
            total_loss += loss * xb.shape[0]

            # backprop
            grad_pred = (2.0 / xb.shape[0]) * err
            gW2 = h.T @ grad_pred
            gb2 = np.sum(grad_pred, axis=0, keepdims=True)

            gh = grad_pred @ W2.T
            gh[h_pre <= 0.0] = 0.0
            gW1 = xb.T @ gh
            gb1 = np.sum(gh, axis=0, keepdims=True)

            W2 -= lr * gW2
            b2 -= lr * gb2
            W1 -= lr * gW1
            b1 -= lr * gb1

        mse = total_loss / n
        print(f"epoch {epoch}/{epochs} mse={mse:.6f}")

    return W1, b1.reshape(-1), W2.reshape(-1), float(b2.reshape(-1)[0])


def quantize_and_save(out_path: Path, W1, b1, W2, b2):
    # Scale for engine integer arithmetic
    s1 = 64.0
    s2 = 64.0
    w1_i = np.clip(np.round(W1.T * s1), -2048, 2047).astype(np.int32)  # [H,768]
    b1_i = np.clip(np.round(b1 * s1), -100000, 100000).astype(np.int32)
    w2_i = np.clip(np.round(W2 * s2), -4096, 4095).astype(np.int32)
    b2_i = int(np.clip(round(b2 * s2), -1000000, 1000000))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("AQUA_NNUE_LITE_V1 768 64\n")
        f.write(" ".join(map(str, b1_i.tolist())) + "\n")
        flat_w1 = w1_i.reshape(-1)
        f.write(" ".join(map(str, flat_w1.tolist())) + "\n")
        f.write(f"{b2_i}\n")
        f.write(" ".join(map(str, w2_i.tolist())) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Train AquaChess NNUE-lite with NumPy")
    ap.add_argument("--npz", type=str, default="", help="Path to .npz containing X [N,768], y [N]")
    ap.add_argument("--pgn", type=str, default="", help="Path to PGN file (requires python-chess)")
    ap.add_argument("--out", type=str, required=True, help="Output weights file path")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-positions", type=int, default=200000)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    if args.hidden != 64:
        raise ValueError("Engine format currently supports hidden=64 only")

    X, y = load_data(args)

    if args.pgn:
        # map game outcome labels to centipawn-ish scale for learning signal
        y = y * 0.8

    print(f"loaded {X.shape[0]} samples")
    W1, b1, W2, b2 = train(X, y, args.hidden, args.epochs, args.lr, args.batch_size, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_and_save(out, W1, b1, W2, b2)
    print(f"saved weights to {out}")


if __name__ == "__main__":
    main()

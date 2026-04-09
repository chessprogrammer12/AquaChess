# AquaChess

AquaChess is a standalone UCI chess engine written in modern C++ with a multi-file layout.

## Features

- Bitboard board representation
- Fully legal move generation (including castling, en passant, and promotions)
- Iterative deepening negamax with alpha-beta pruning
- Quiescence search
- Transposition table
- MVV-LVA capture ordering, killer moves, and history heuristic
- Basic tapered positional evaluation with piece-square tables, mobility, pawn structure, bishop pair, rook files, and king safety heuristics
- UCI protocol support
- Multi-file architecture (`src/main.cpp`, `src/engine.cpp`, `include/aqua/engine.hpp`)
- Added strength-oriented pruning/search improvements (null-move pruning and stronger late-move reductions)

## Build

```bash
make
```

## Run

```bash
./aquachess
```

Then talk to it over UCI, for example:

```text
uci
isready
ucinewgame
position startpos moves e2e4 e7e5
go depth 8
```

Or from a shell, to verify the handshake explicitly:

```bash
printf 'uci\nisready\nquit\n' | ./aquachess
```

### Optional NNUE-like eval file

You can enable an external neural eval file through UCI options:

```text
setoption name EvalFile value /path/to/weights.nnuelite
setoption name UseNNUE value true
```

Current file format is a simple text format:

1. Header line: `AQUA_NNUE_LITE_V1 768 64`
2. 64 hidden biases
3. 64x768 input-to-hidden integer weights
4. One output bias
5. 64 hidden-to-output integer weights

This lets you train weights locally from your own Lichess PGN/database and plug them into AquaChess.

### Train your own NNUE-lite with NumPy

I added a NumPy trainer script at `tools/train_nnue_numpy.py` so you can train on your own laptop data:

```bash
python tools/train_nnue_numpy.py --pgn /path/to/lichess_games.pgn --out weights.nnuelite --epochs 8
```

Or train from a prepared NPZ dataset:

```bash
python tools/train_nnue_numpy.py --npz /path/to/train_data.npz --out weights.nnuelite --epochs 8
```

Then load in engine:

```text
setoption name EvalFile value /absolute/path/to/weights.nnuelite
setoption name UseNNUE value true
```

## Notes on strength

This engine implements the main architecture used by many classical engines, but Elo depends heavily on tuning, hardware, testing pool, opening books, endgame tablebases, and long-term parameter optimization. It should be meaningfully stronger than a toy engine, but no honest engine author can guarantee 2300-2700 Elo without extensive benchmarking and tuning.

## About strength vs 2000 Elo

With stronger search defaults and this engine architecture, you may beat some ~2000-strength bots in practical games depending on time control and hardware. But Elo is environment-dependent, so this still requires testing and tuning to verify consistent results against a specific 2000 Elo engine.

For 2300-2700 bots, expect that you will still need serious tuning (more search work, larger tests, and stronger trained eval data) to be consistently competitive.

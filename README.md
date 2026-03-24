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

## Notes on strength

This engine implements the main architecture used by many classical engines, but Elo depends heavily on tuning, hardware, testing pool, opening books, endgame tablebases, and long-term parameter optimization. It should be meaningfully stronger than a toy engine, but no honest engine author can guarantee 2300-2700 Elo without extensive benchmarking and tuning.

## About strength vs 2000 Elo

With stronger search defaults and this engine architecture, you may beat some ~2000-strength bots in practical games depending on time control and hardware. But Elo is environment-dependent, so this still requires testing and tuning to verify consistent results against a specific 2000 Elo engine.

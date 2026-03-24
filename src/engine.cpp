#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>


#include "aqua/engine.hpp"
namespace aqua {

using Bitboard = std::uint64_t;
using Score = int;

constexpr Score INF = 30000;
constexpr Score MATE = 29000;
constexpr int MAX_PLY = 128;
constexpr int TT_SIZE_MB = 64;
constexpr int TT_ENTRIES = (TT_SIZE_MB * 1024 * 1024) / static_cast<int>(sizeof(std::uint64_t) * 4);

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NB, NO_PIECE_TYPE = 7 };
enum Piece : int {
    W_PAWN = 0,
    W_KNIGHT,
    W_BISHOP,
    W_ROOK,
    W_QUEEN,
    W_KING,
    B_PAWN,
    B_KNIGHT,
    B_BISHOP,
    B_ROOK,
    B_QUEEN,
    B_KING,
    NO_PIECE = 15
};

enum MoveFlags : std::uint8_t {
    QUIET = 0,
    DOUBLE_PUSH = 1,
    KING_CASTLE = 2,
    QUEEN_CASTLE = 3,
    CAPTURE = 4,
    EN_PASSANT = 5,
    PROMOTION = 8
};

constexpr std::array<char, 12> PIECE_TO_CHAR = {'P','N','B','R','Q','K','p','n','b','r','q','k'};
constexpr std::array<int, PIECE_TYPE_NB> PIECE_VALUES = {100, 320, 330, 500, 900, 0};
constexpr std::array<int, PIECE_TYPE_NB> GAMEPHASE_INC = {0, 1, 1, 2, 4, 0};

constexpr std::array<int, 64> PAWN_PST = {
      0,  0,  0,  0,  0,  0,  0,  0,
     70, 75, 75, 80, 80, 75, 75, 70,
     30, 35, 40, 55, 55, 40, 35, 30,
     10, 15, 20, 35, 35, 20, 15, 10,
      5, 10, 10, 25, 25, 10, 10,  5,
      5,  0,  0, 10, 10,  0,  0,  5,
      5,  5,  5,-20,-20,  5,  5,  5,
      0,  0,  0,  0,  0,  0,  0,  0
};
constexpr std::array<int, 64> KNIGHT_PST = {
    -70,-40,-30,-25,-25,-30,-40,-70,
    -40,-10,  0,  5,  5,  0,-10,-40,
    -25, 10, 18, 20, 20, 18, 10,-25,
    -20, 10, 25, 30, 30, 25, 10,-20,
    -20, 10, 25, 30, 30, 25, 10,-20,
    -25, 10, 18, 20, 20, 18, 10,-25,
    -40,-10,  0,  5,  5,  0,-10,-40,
    -70,-40,-25,-20,-20,-25,-40,-70
};
constexpr std::array<int, 64> BISHOP_PST = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  5,  5,  0,  5,-10,
    -10, 10, 10, 15, 15, 10, 10,-10,
    -10,  0, 15, 20, 20, 15,  0,-10,
    -10,  5, 15, 20, 20, 15,  5,-10,
    -10,  0, 10, 15, 15, 10,  0,-10,
    -10,  0,  0,  5,  5,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};
constexpr std::array<int, 64> ROOK_PST = {
      0,  0,  5, 10, 10,  5,  0,  0,
     -5,  0,  0,  5,  5,  0,  0, -5,
     -5,  0,  0,  5,  5,  0,  0, -5,
     -5,  0,  0,  5,  5,  0,  0, -5,
     -5,  0,  0,  5,  5,  0,  0, -5,
     -5,  0,  0,  5,  5,  0,  0, -5,
      5, 10, 10, 10, 10, 10, 10,  5,
      0,  0,  5, 10, 10,  5,  0,  0
};
constexpr std::array<int, 64> QUEEN_PST = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};
constexpr std::array<int, 64> KING_MID_PST = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 25, 25, 20,-10,-30,
    -30,-10, 25, 30, 30, 25,-10,-30,
    -30,-10, 25, 30, 30, 25,-10,-30,
    -30,-10, 20, 25, 25, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};
constexpr std::array<int, 64> KING_END_PST = {
    -40,-30,-20,-10,-10,-20,-30,-40,
    -30,-15, -5,  0,  0, -5,-15,-30,
    -20, -5, 15, 20, 20, 15, -5,-20,
    -10,  0, 20, 30, 30, 20,  0,-10,
    -10,  0, 20, 30, 30, 20,  0,-10,
    -20, -5, 15, 20, 20, 15, -5,-20,
    -30,-15, -5,  0,  0, -5,-15,-30,
    -40,-30,-20,-10,-10,-20,-30,-40
};

constexpr std::array<Bitboard, 8> FILE_MASKS = [] {
    std::array<Bitboard, 8> arr{};
    for (int file = 0; file < 8; ++file) {
        Bitboard mask = 0;
        for (int rank = 0; rank < 8; ++rank) {
            mask |= 1ULL << (rank * 8 + file);
        }
        arr[file] = mask;
    }
    return arr;
}();

constexpr std::array<Bitboard, 8> RANK_MASKS = [] {
    std::array<Bitboard, 8> arr{};
    for (int rank = 0; rank < 8; ++rank) {
        arr[rank] = 0xFFULL << (rank * 8);
    }
    return arr;
}();

constexpr int mirror_sq(int sq) {
    return sq ^ 56;
}

constexpr Bitboard bit(int sq) {
    return 1ULL << sq;
}

inline int file_of(int sq) { return sq & 7; }
inline int rank_of(int sq) { return sq >> 3; }
inline int pop_lsb(Bitboard &bb) {
    const int sq = std::countr_zero(bb);
    bb &= bb - 1;
    return sq;
}
inline int popcount(Bitboard bb) { return std::popcount(bb); }

struct Move {
    std::uint32_t raw = 0;
    Move() = default;
    Move(int from, int to, int flags = QUIET, int promo = NO_PIECE_TYPE) {
        raw = static_cast<std::uint32_t>((from & 63) | ((to & 63) << 6) | ((promo & 3) << 12) | ((flags & 15) << 14));
    }
    int from() const { return raw & 63; }
    int to() const { return (raw >> 6) & 63; }
    int promo() const { return (raw >> 12) & 3; }
    int flags() const { return (raw >> 14) & 15; }
    bool is_promotion() const { return flags() & PROMOTION; }
    bool is_capture() const { return flags() == CAPTURE || flags() == EN_PASSANT || ((flags() & 4) != 0); }
    bool operator==(const Move &other) const = default;
};

std::string move_to_uci(Move move) {
    const char files[] = "abcdefgh";
    std::string out;
    out += files[file_of(move.from())];
    out += static_cast<char>('1' + rank_of(move.from()));
    out += files[file_of(move.to())];
    out += static_cast<char>('1' + rank_of(move.to()));
    if (move.is_promotion()) {
        const char promo_chars[] = {'n', 'b', 'r', 'q'};
        out += promo_chars[move.promo()];
    }
    return out;
}

struct Undo {
    int captured = NO_PIECE;
    int castling = 0;
    int ep = -1;
    int halfmove = 0;
    std::uint64_t key = 0;
};

struct Zobrist {
    std::uint64_t piece[12][64]{};
    std::uint64_t castling[16]{};
    std::uint64_t ep[8]{};
    std::uint64_t side = 0;
    Zobrist() {
        std::mt19937_64 rng(0xA11CEBADA55ULL);
        for (auto &arr : piece) for (auto &x : arr) x = rng();
        for (auto &x : castling) x = rng();
        for (auto &x : ep) x = rng();
        side = rng();
    }
};

Zobrist ZB;
std::array<Bitboard, 64> KNIGHT_ATTACKS{};
std::array<Bitboard, 64> KING_ATTACKS{};
std::array<std::array<Bitboard, 64>, 2> PAWN_ATTACKS{};

Bitboard sliding_attacks(int sq, Bitboard occ, const std::vector<int> &dirs) {
    Bitboard attacks = 0;
    for (int dir : dirs) {
        int s = sq;
        while (true) {
            const int next = s + dir;
            if (next < 0 || next >= 64) break;
            if (std::abs(file_of(next) - file_of(s)) > 1 && (dir == 1 || dir == -1 || dir == 9 || dir == -9 || dir == 7 || dir == -7)) break;
            s = next;
            attacks |= bit(s);
            if (occ & bit(s)) break;
        }
    }
    return attacks;
}

struct Position {
    std::array<Bitboard, 12> piece_bb{};
    std::array<Bitboard, 3> occ{};
    std::array<int, 64> board{};
    Color side_to_move = WHITE;
    int castling = 0;
    int ep_square = -1;
    int halfmove = 0;
    int fullmove = 1;
    std::uint64_t key = 0;

    Position() { board.fill(NO_PIECE); }

    void clear() {
        piece_bb.fill(0);
        occ.fill(0);
        board.fill(NO_PIECE);
        side_to_move = WHITE;
        castling = 0;
        ep_square = -1;
        halfmove = 0;
        fullmove = 1;
        key = 0;
    }

    static int piece_color(int p) { return p >= B_PAWN ? BLACK : WHITE; }
    static PieceType piece_type(int p) { return p == NO_PIECE ? NO_PIECE_TYPE : static_cast<PieceType>(p % 6); }

    void put_piece(int piece, int sq) {
        piece_bb[piece] |= bit(sq);
        occ[piece_color(piece)] |= bit(sq);
        occ[2] |= bit(sq);
        board[sq] = piece;
        key ^= ZB.piece[piece][sq];
    }

    void remove_piece(int piece, int sq) {
        piece_bb[piece] &= ~bit(sq);
        occ[piece_color(piece)] &= ~bit(sq);
        occ[2] &= ~bit(sq);
        board[sq] = NO_PIECE;
        key ^= ZB.piece[piece][sq];
    }

    void move_piece(int piece, int from, int to) {
        piece_bb[piece] ^= bit(from) | bit(to);
        occ[piece_color(piece)] ^= bit(from) | bit(to);
        occ[2] ^= bit(from) | bit(to);
        board[from] = NO_PIECE;
        board[to] = piece;
        key ^= ZB.piece[piece][from] ^ ZB.piece[piece][to];
    }

    void set_fen(const std::string &fen) {
        clear();
        std::istringstream iss(fen);
        std::string board_part, side_part, castling_part, ep_part;
        iss >> board_part >> side_part >> castling_part >> ep_part >> halfmove >> fullmove;
        int sq = 56;
        for (char c : board_part) {
            if (c == '/') {
                sq -= 16;
            } else if (std::isdigit(static_cast<unsigned char>(c))) {
                sq += c - '0';
            } else {
                auto it = std::find(PIECE_TO_CHAR.begin(), PIECE_TO_CHAR.end(), c);
                if (it != PIECE_TO_CHAR.end()) {
                    put_piece(static_cast<int>(it - PIECE_TO_CHAR.begin()), sq);
                }
                ++sq;
            }
        }
        side_to_move = (side_part == "w") ? WHITE : BLACK;
        if (side_to_move == BLACK) key ^= ZB.side;
        castling = 0;
        if (castling_part.find('K') != std::string::npos) castling |= 1;
        if (castling_part.find('Q') != std::string::npos) castling |= 2;
        if (castling_part.find('k') != std::string::npos) castling |= 4;
        if (castling_part.find('q') != std::string::npos) castling |= 8;
        key ^= ZB.castling[castling];
        ep_square = -1;
        if (ep_part != "-") {
            ep_square = (ep_part[0] - 'a') + (ep_part[1] - '1') * 8;
            key ^= ZB.ep[file_of(ep_square)];
        }
    }

    bool square_attacked(int sq, Color by) const {
        if (PAWN_ATTACKS[by ^ 1][sq] & piece_bb[(by == WHITE) ? W_PAWN : B_PAWN]) return true;
        if (KNIGHT_ATTACKS[sq] & piece_bb[(by == WHITE) ? W_KNIGHT : B_KNIGHT]) return true;
        if (KING_ATTACKS[sq] & piece_bb[(by == WHITE) ? W_KING : B_KING]) return true;
        const Bitboard bishops = piece_bb[(by == WHITE) ? W_BISHOP : B_BISHOP] | piece_bb[(by == WHITE) ? W_QUEEN : B_QUEEN];
        const Bitboard rooks = piece_bb[(by == WHITE) ? W_ROOK : B_ROOK] | piece_bb[(by == WHITE) ? W_QUEEN : B_QUEEN];
        if (sliding_attacks(sq, occ[2], {9, 7, -7, -9}) & bishops) return true;
        if (sliding_attacks(sq, occ[2], {8, -8, 1, -1}) & rooks) return true;
        return false;
    }

    int king_square(Color c) const {
        Bitboard bb = piece_bb[c == WHITE ? W_KING : B_KING];
        return bb ? std::countr_zero(bb) : -1;
    }

    bool in_check(Color c) const {
        return square_attacked(king_square(c), static_cast<Color>(c ^ 1));
    }

    bool make_move(Move m, Undo &u) {
        u.captured = NO_PIECE;
        u.castling = castling;
        u.ep = ep_square;
        u.halfmove = halfmove;
        u.key = key;

        const int from = m.from();
        const int to = m.to();
        const int piece = board[from];
        const int flags = m.flags();
        const Color us = side_to_move;
        const Color them = static_cast<Color>(us ^ 1);

        key ^= ZB.castling[castling];
        if (ep_square != -1) key ^= ZB.ep[file_of(ep_square)];
        ep_square = -1;

        if (flags == EN_PASSANT) {
            const int cap_sq = to + (us == WHITE ? -8 : 8);
            u.captured = board[cap_sq];
            remove_piece(u.captured, cap_sq);
        } else if (board[to] != NO_PIECE) {
            u.captured = board[to];
            remove_piece(u.captured, to);
        }

        if (Position::piece_type(piece) == KING) {
            castling &= (us == WHITE) ? ~3 : ~12;
            if (flags == KING_CASTLE) {
                move_piece(piece, from, to);
                move_piece(us == WHITE ? W_ROOK : B_ROOK, to + 1, to - 1);
            } else if (flags == QUEEN_CASTLE) {
                move_piece(piece, from, to);
                move_piece(us == WHITE ? W_ROOK : B_ROOK, to - 2, to + 1);
            } else {
                move_piece(piece, from, to);
            }
        } else {
            if (from == 0 || to == 0) castling &= ~2;
            if (from == 7 || to == 7) castling &= ~1;
            if (from == 56 || to == 56) castling &= ~8;
            if (from == 63 || to == 63) castling &= ~4;
            move_piece(piece, from, to);
            if (Position::piece_type(piece) == PAWN) {
                halfmove = 0;
                if (flags == DOUBLE_PUSH) ep_square = to + (us == WHITE ? -8 : 8);
                if (m.is_promotion()) {
                    remove_piece(piece, to);
                    const int promoted = (us == WHITE ? W_KNIGHT : B_KNIGHT) + m.promo();
                    put_piece(promoted, to);
                }
            }
        }

        if (u.captured != NO_PIECE || Position::piece_type(piece) == PAWN) halfmove = 0;
        else ++halfmove;

        key ^= ZB.castling[castling];
        if (ep_square != -1) key ^= ZB.ep[file_of(ep_square)];
        side_to_move = them;
        key ^= ZB.side;
        if (us == BLACK) ++fullmove;

        if (in_check(us)) {
            unmake_move(m, u);
            return false;
        }
        return true;
    }

    void unmake_move(Move m, const Undo &u) {
        const int from = m.from();
        const int to = m.to();
        side_to_move = static_cast<Color>(side_to_move ^ 1);
        if (side_to_move == BLACK) --fullmove;

        key = u.key;
        castling = u.castling;
        ep_square = u.ep;
        halfmove = u.halfmove;

        const Color us = side_to_move;
        int piece = board[to];

        if (m.is_promotion()) {
            remove_piece(piece, to);
            piece = (us == WHITE) ? W_PAWN : B_PAWN;
            put_piece(piece, to);
        }

        if (m.flags() == KING_CASTLE) {
            move_piece(piece, to, from);
            move_piece(us == WHITE ? W_ROOK : B_ROOK, to - 1, to + 1);
        } else if (m.flags() == QUEEN_CASTLE) {
            move_piece(piece, to, from);
            move_piece(us == WHITE ? W_ROOK : B_ROOK, to + 1, to - 2);
        } else {
            move_piece(piece, to, from);
            if (m.flags() == EN_PASSANT) {
                const int cap_sq = to + (us == WHITE ? -8 : 8);
                put_piece(u.captured, cap_sq);
            } else if (u.captured != NO_PIECE) {
                put_piece(u.captured, to);
            }
        }
    }
};

struct ScoredMove { Move move; int score; };

struct TTEntry {
    std::uint64_t key = 0;
    std::uint32_t move = 0;
    std::int16_t score = 0;
    std::int8_t depth = -1;
    std::uint8_t flag = 0;
};

enum TTFlag : std::uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };

struct TranspositionTable {
    std::vector<TTEntry> table;
    TranspositionTable() : table(static_cast<std::size_t>(TT_ENTRIES)) {}

    TTEntry *probe(std::uint64_t key) {
        return &table[key % table.size()];
    }

    void store(std::uint64_t key, int depth, int flag, int score, Move move) {
        TTEntry *e = probe(key);
        if (e->depth > depth && e->key != key) return;
        e->key = key;
        e->depth = static_cast<std::int8_t>(depth);
        e->flag = static_cast<std::uint8_t>(flag);
        e->score = static_cast<std::int16_t>(score);
        e->move = move.raw;
    }
};

struct Searcher {
    TranspositionTable tt;
    std::array<std::array<int, 64>, 64> history{};
    std::array<std::array<Move, 2>, MAX_PLY> killers{};
    std::chrono::steady_clock::time_point stop_time{};
    std::atomic<bool> stopped = false;
    std::uint64_t nodes = 0;
    Move best_root_move{};

    bool time_up() {
        return std::chrono::steady_clock::now() >= stop_time;
    }

    static int pst_value(PieceType pt, int sq, Color c, int phase_endgame) {
        const int idx = (c == WHITE) ? mirror_sq(sq) : sq;
        switch (pt) {
            case PAWN: return PAWN_PST[idx];
            case KNIGHT: return KNIGHT_PST[idx];
            case BISHOP: return BISHOP_PST[idx];
            case ROOK: return ROOK_PST[idx];
            case QUEEN: return QUEEN_PST[idx];
            case KING: return phase_endgame ? KING_END_PST[idx] : KING_MID_PST[idx];
            default: return 0;
        }
    }

    static Bitboard bishop_attacks(int sq, Bitboard occ) { return sliding_attacks(sq, occ, {9, 7, -7, -9}); }
    static Bitboard rook_attacks(int sq, Bitboard occ) { return sliding_attacks(sq, occ, {8, -8, 1, -1}); }

    int evaluate(const Position &pos) {
        int mg[2] = {0, 0};
        int eg[2] = {0, 0};
        int phase = 0;

        for (int piece = 0; piece < 12; ++piece) {
            Bitboard bb = pos.piece_bb[piece];
            const Color c = static_cast<Color>(piece / 6);
            const PieceType pt = static_cast<PieceType>(piece % 6);
            while (bb) {
                const int sq = pop_lsb(bb);
                mg[c] += PIECE_VALUES[pt] + pst_value(pt, sq, c, 0);
                eg[c] += PIECE_VALUES[pt] + pst_value(pt, sq, c, 1);
                phase += GAMEPHASE_INC[pt];

                if (pt == BISHOP) {
                    mg[c] += popcount(bishop_attacks(sq, pos.occ[2]) & ~pos.occ[c]) * 3;
                    eg[c] += popcount(bishop_attacks(sq, pos.occ[2]) & ~pos.occ[c]) * 4;
                } else if (pt == ROOK) {
                    const int f = file_of(sq);
                    const bool open = (pos.piece_bb[W_PAWN] | pos.piece_bb[B_PAWN]) & FILE_MASKS[f] ? false : true;
                    const bool semi = (c == WHITE ? pos.piece_bb[W_PAWN] : pos.piece_bb[B_PAWN]) & FILE_MASKS[f] ? false : true;
                    mg[c] += open ? 18 : (semi ? 9 : 0);
                    mg[c] += popcount(rook_attacks(sq, pos.occ[2]) & ~pos.occ[c]) * 2;
                    eg[c] += open ? 12 : (semi ? 6 : 0);
                } else if (pt == QUEEN) {
                    mg[c] += popcount((rook_attacks(sq, pos.occ[2]) | bishop_attacks(sq, pos.occ[2])) & ~pos.occ[c]);
                } else if (pt == KNIGHT) {
                    mg[c] += popcount(KNIGHT_ATTACKS[sq] & ~pos.occ[c]) * 3;
                    eg[c] += popcount(KNIGHT_ATTACKS[sq] & ~pos.occ[c]) * 2;
                }
            }
        }

        for (int c = 0; c < 2; ++c) {
            const Bitboard pawns = pos.piece_bb[c == WHITE ? W_PAWN : B_PAWN];
            Bitboard bb = pawns;
            std::array<int, 8> file_count{};
            Bitboard tmp = pawns;
            while (tmp) file_count[file_of(pop_lsb(tmp))]++;
            while (bb) {
                const int sq = pop_lsb(bb);
                const int f = file_of(sq);
                if (file_count[f] > 1) { mg[c] -= 10; eg[c] -= 12; }
                const Bitboard neighbors = (f > 0 ? FILE_MASKS[f - 1] : 0) | (f < 7 ? FILE_MASKS[f + 1] : 0);
                if ((pawns & neighbors) == 0) { mg[c] -= 12; eg[c] -= 15; }
                const Bitboard enemy_pawns = pos.piece_bb[c == WHITE ? B_PAWN : W_PAWN];
                Bitboard blockers = enemy_pawns & ((c == WHITE) ? (~0ULL << (sq + 8)) : ((1ULL << sq) - 1));
                Bitboard enemy_neighbor_files = (f > 0 ? FILE_MASKS[f - 1] : 0) | FILE_MASKS[f] | (f < 7 ? FILE_MASKS[f + 1] : 0);
                if ((blockers & enemy_neighbor_files) == 0) {
                    const int advance = c == WHITE ? rank_of(sq) : 7 - rank_of(sq);
                    mg[c] += 8 * advance;
                    eg[c] += 18 * advance;
                }
            }

            if (popcount(pos.piece_bb[c == WHITE ? W_BISHOP : B_BISHOP]) >= 2) {
                mg[c] += 30;
                eg[c] += 45;
            }

            const int king_sq = pos.king_square(static_cast<Color>(c));
            const int kf = file_of(king_sq);
            const int shield_rank = c == WHITE ? 1 : 6;
            int shield = 0;
            for (int df = -1; df <= 1; ++df) {
                const int f = kf + df;
                if (f >= 0 && f < 8) {
                    const int sq = shield_rank * 8 + f;
                    if (pos.board[sq] == (c == WHITE ? W_PAWN : B_PAWN)) shield += 10;
                }
            }
            mg[c] += shield;
        }

        phase = std::min(phase, 24);
        const int mg_score = mg[WHITE] - mg[BLACK];
        const int eg_score = eg[WHITE] - eg[BLACK];
        int score = (mg_score * phase + eg_score * (24 - phase)) / 24;
        return pos.side_to_move == WHITE ? score : -score;
    }

    int score_move(const Position &pos, Move m, int ply, Move tt_move) {
        if (m == tt_move) return 2'000'000;
        const int from_piece = pos.board[m.from()];
        if (m.is_capture()) {
            int victim_piece = pos.board[m.to()];
            if (m.flags() == EN_PASSANT) victim_piece = pos.side_to_move == WHITE ? B_PAWN : W_PAWN;
            const int attacker = Position::piece_type(from_piece);
            const int victim = Position::piece_type(victim_piece);
            return 1'000'000 + 16 * PIECE_VALUES[victim] - PIECE_VALUES[attacker];
        }
        if (killers[ply][0] == m) return 900'000;
        if (killers[ply][1] == m) return 800'000;
        return history[m.from()][m.to()];
    }

    std::vector<ScoredMove> generate_moves(const Position &pos, bool captures_only = false) {
        std::vector<ScoredMove> moves;
        const Color us = pos.side_to_move;
        const Color them = static_cast<Color>(us ^ 1);
        const Bitboard own = pos.occ[us];
        const Bitboard enemy = pos.occ[them];
        const int pawn_piece = (us == WHITE) ? W_PAWN : B_PAWN;
        const int knight_piece = (us == WHITE) ? W_KNIGHT : B_KNIGHT;
        const int bishop_piece = (us == WHITE) ? W_BISHOP : B_BISHOP;
        const int rook_piece = (us == WHITE) ? W_ROOK : B_ROOK;
        const int queen_piece = (us == WHITE) ? W_QUEEN : B_QUEEN;
        const int king_piece = (us == WHITE) ? W_KING : B_KING;

        Bitboard pawns = pos.piece_bb[pawn_piece];
        while (pawns) {
            int from = pop_lsb(pawns);
            const int rank = rank_of(from);
            const int dir = us == WHITE ? 8 : -8;
            const int one = from + dir;
            if (!captures_only && one >= 0 && one < 64 && pos.board[one] == NO_PIECE) {
                if (rank == (us == WHITE ? 6 : 1)) {
                    for (int promo = 0; promo < 4; ++promo) moves.push_back({Move(from, one, PROMOTION, promo), 0});
                } else {
                    moves.push_back({Move(from, one), 0});
                    const int two = from + dir * 2;
                    if ((rank == 1 && us == WHITE) || (rank == 6 && us == BLACK)) {
                        if (pos.board[two] == NO_PIECE) moves.push_back({Move(from, two, DOUBLE_PUSH), 0});
                    }
                }
            }
            Bitboard caps = PAWN_ATTACKS[us][from] & enemy;
            while (caps) {
                int to = pop_lsb(caps);
                if (rank == (us == WHITE ? 6 : 1)) {
                    for (int promo = 0; promo < 4; ++promo) moves.push_back({Move(from, to, CAPTURE | PROMOTION, promo), 0});
                } else {
                    moves.push_back({Move(from, to, CAPTURE), 0});
                }
            }
            if (pos.ep_square != -1 && (PAWN_ATTACKS[us][from] & bit(pos.ep_square))) {
                moves.push_back({Move(from, pos.ep_square, EN_PASSANT), 0});
            }
        }

        auto add_piece_moves = [&](Bitboard bb, auto attacks_fn) {
            while (bb) {
                const int from = pop_lsb(bb);
                Bitboard targets = attacks_fn(from) & ~own;
                if (captures_only) targets &= enemy;
                while (targets) {
                    const int to = pop_lsb(targets);
                    moves.push_back({Move(from, to, pos.board[to] == NO_PIECE ? QUIET : CAPTURE), 0});
                }
            }
        };

        add_piece_moves(pos.piece_bb[knight_piece], [&](int from) { return KNIGHT_ATTACKS[from]; });
        add_piece_moves(pos.piece_bb[bishop_piece], [&](int from) { return bishop_attacks(from, pos.occ[2]); });
        add_piece_moves(pos.piece_bb[rook_piece], [&](int from) { return rook_attacks(from, pos.occ[2]); });
        add_piece_moves(pos.piece_bb[queen_piece], [&](int from) { return bishop_attacks(from, pos.occ[2]) | rook_attacks(from, pos.occ[2]); });
        add_piece_moves(pos.piece_bb[king_piece], [&](int from) { return KING_ATTACKS[from]; });

        if (!captures_only) {
            if (us == WHITE && !pos.in_check(WHITE)) {
                if ((pos.castling & 1) && pos.board[5] == NO_PIECE && pos.board[6] == NO_PIECE && !pos.square_attacked(5, BLACK) && !pos.square_attacked(6, BLACK)) {
                    moves.push_back({Move(4, 6, KING_CASTLE), 0});
                }
                if ((pos.castling & 2) && pos.board[1] == NO_PIECE && pos.board[2] == NO_PIECE && pos.board[3] == NO_PIECE && !pos.square_attacked(2, BLACK) && !pos.square_attacked(3, BLACK)) {
                    moves.push_back({Move(4, 2, QUEEN_CASTLE), 0});
                }
            } else if (us == BLACK && !pos.in_check(BLACK)) {
                if ((pos.castling & 4) && pos.board[61] == NO_PIECE && pos.board[62] == NO_PIECE && !pos.square_attacked(61, WHITE) && !pos.square_attacked(62, WHITE)) {
                    moves.push_back({Move(60, 62, KING_CASTLE), 0});
                }
                if ((pos.castling & 8) && pos.board[57] == NO_PIECE && pos.board[58] == NO_PIECE && pos.board[59] == NO_PIECE && !pos.square_attacked(58, WHITE) && !pos.square_attacked(59, WHITE)) {
                    moves.push_back({Move(60, 58, QUEEN_CASTLE), 0});
                }
            }
        }
        return moves;
    }

    int quiescence(Position &pos, int alpha, int beta, int ply) {
        if ((nodes & 2047ULL) == 0 && time_up()) { stopped = true; return 0; }
        ++nodes;
        int stand_pat = evaluate(pos);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;

        auto moves = generate_moves(pos, true);
        for (auto &sm : moves) sm.score = score_move(pos, sm.move, ply, Move{});
        std::sort(moves.begin(), moves.end(), [](const ScoredMove &a, const ScoredMove &b) { return a.score > b.score; });

        for (const auto &sm : moves) {
            Undo u;
            if (!pos.make_move(sm.move, u)) continue;
            int score = -quiescence(pos, -beta, -alpha, ply + 1);
            pos.unmake_move(sm.move, u);
            if (stopped) return 0;
            if (score >= beta) return beta;
            alpha = std::max(alpha, score);
        }
        return alpha;
    }

    int negamax(Position &pos, int depth, int alpha, int beta, int ply) {
        if ((nodes & 2047ULL) == 0 && time_up()) { stopped = true; return 0; }
        ++nodes;
        const bool root = ply == 0;
        const bool in_check = pos.in_check(pos.side_to_move);
        if (depth <= 0) return quiescence(pos, alpha, beta, ply);
        if (ply >= MAX_PLY - 1) return evaluate(pos);

        const int original_alpha = alpha;
        TTEntry *entry = tt.probe(pos.key);
        Move tt_move{};
        if (entry->key == pos.key) {
            tt_move.raw = entry->move;
            if (entry->depth >= depth) {
                const int tt_score = entry->score;
                if (entry->flag == TT_EXACT) return tt_score;
                if (entry->flag == TT_ALPHA && tt_score <= alpha) return tt_score;
                if (entry->flag == TT_BETA && tt_score >= beta) return tt_score;
            }
        }

        auto moves = generate_moves(pos, false);
        for (auto &sm : moves) sm.score = score_move(pos, sm.move, ply, tt_move);
        std::sort(moves.begin(), moves.end(), [](const ScoredMove &a, const ScoredMove &b) { return a.score > b.score; });

        if (moves.empty()) {
            return in_check ? -MATE + ply : 0;
        }

        Move best_move{};
        int best_score = -INF;
        int legal_moves = 0;

        for (std::size_t i = 0; i < moves.size(); ++i) {
            Undo u;
            Move move = moves[i].move;
            if (!pos.make_move(move, u)) continue;
            ++legal_moves;
            int score;
            if (i > 0 && depth >= 3 && !in_check && !move.is_capture() && !move.is_promotion()) {
                score = -negamax(pos, depth - 2, -alpha - 1, -alpha, ply + 1);
                if (score > alpha) score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            } else {
                score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            }
            pos.unmake_move(move, u);
            if (stopped) return 0;
            if (score > best_score) {
                best_score = score;
                best_move = move;
                if (root) best_root_move = move;
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                if (!move.is_capture()) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = move;
                    history[move.from()][move.to()] += depth * depth;
                }
                break;
            }
        }

        if (legal_moves == 0) return in_check ? -MATE + ply : 0;

        int flag = TT_EXACT;
        if (best_score <= original_alpha) flag = TT_ALPHA;
        else if (best_score >= beta) flag = TT_BETA;
        tt.store(pos.key, depth, flag, best_score, best_move);
        return best_score;
    }

    Move search(Position &pos, int max_depth, int time_ms) {
        stopped = false;
        nodes = 0;
        best_root_move = Move{};
        stop_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_ms);
        int score = 0;
        int alpha = -INF;
        int beta = INF;

        for (int depth = 1; depth <= max_depth; ++depth) {
            const int window = depth > 4 ? 40 : INF;
            if (window != INF) {
                alpha = score - window;
                beta = score + window;
            } else {
                alpha = -INF;
                beta = INF;
            }
            while (true) {
                score = negamax(pos, depth, alpha, beta, 0);
                if (stopped) break;
                if (score <= alpha) {
                    alpha -= window == INF ? INF / 2 : window * 2;
                } else if (score >= beta) {
                    beta += window == INF ? INF / 2 : window * 2;
                } else {
                    break;
                }
            }
            if (stopped) break;
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - (stop_time - std::chrono::milliseconds(time_ms))).count();
            std::cout << "info depth " << depth << " score cp " << score << " nodes " << nodes << " time " << elapsed << " pv " << move_to_uci(best_root_move) << std::endl;
        }
        return best_root_move;
    }
};

void init_tables() {
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard attacks = 0;
        const int r = rank_of(sq);
        const int f = file_of(sq);
        for (const auto &[dr, df] : std::array<std::pair<int, int>, 8>{{{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}}) {
            const int nr = r + dr;
            const int nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) attacks |= bit(nr * 8 + nf);
        }
        KNIGHT_ATTACKS[sq] = attacks;
        attacks = 0;
        for (const auto &[dr, df] : std::array<std::pair<int, int>, 8>{{{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}}) {
            const int nr = r + dr;
            const int nf = f + df;
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) attacks |= bit(nr * 8 + nf);
        }
        KING_ATTACKS[sq] = attacks;
        Bitboard white = 0;
        if (r < 7 && f > 0) white |= bit((r + 1) * 8 + (f - 1));
        if (r < 7 && f < 7) white |= bit((r + 1) * 8 + (f + 1));
        PAWN_ATTACKS[WHITE][sq] = white;
        Bitboard black = 0;
        if (r > 0 && f > 0) black |= bit((r - 1) * 8 + (f - 1));
        if (r > 0 && f < 7) black |= bit((r - 1) * 8 + (f + 1));
        PAWN_ATTACKS[BLACK][sq] = black;
    }
}

std::vector<std::string> split(const std::string &s) {
    std::istringstream iss(s);
    std::vector<std::string> tokens;
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

constexpr const char *STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

Move parse_uci_move(Position &pos, const std::string &text, Searcher &searcher) {
    auto moves = searcher.generate_moves(pos, false);
    for (const auto &sm : moves) {
        Undo u;
        if (!pos.make_move(sm.move, u)) continue;
        pos.unmake_move(sm.move, u);
        if (move_to_uci(sm.move) == text) return sm.move;
    }
    return Move{};
}

void reset_searcher(Searcher &searcher) {
    searcher.tt = TranspositionTable{};
    searcher.history = {};
    searcher.killers = {};
    searcher.stopped = false;
    searcher.nodes = 0;
    searcher.best_root_move = Move{};
}

bool apply_position_command(Position &pos, Searcher &searcher, const std::vector<std::string> &tokens) {
    if (tokens.size() < 2) return false;

    std::size_t idx = 1;
    if (tokens[idx] == "startpos") {
        pos.set_fen(STARTPOS_FEN);
        ++idx;
    } else if (tokens[idx] == "fen") {
        if (tokens.size() < idx + 7) return false;
        std::string fen;
        for (int i = 0; i < 6; ++i) {
            if (i != 0) fen += ' ';
            fen += tokens[idx + 1 + static_cast<std::size_t>(i)];
        }
        pos.set_fen(fen);
        idx += 7;
    } else {
        return false;
    }

    if (idx < tokens.size() && tokens[idx] == "moves") {
        for (++idx; idx < tokens.size(); ++idx) {
            Move m = parse_uci_move(pos, tokens[idx], searcher);
            if (m.raw == 0) return false;
            Undo u;
            if (!pos.make_move(m, u)) return false;
        }
    }
    return true;
}

} // namespace aqua

namespace aqua {

int run_uci() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.setf(std::ios::unitbuf);

    init_tables();
    Position pos;
    pos.set_fen(STARTPOS_FEN);
    Searcher searcher;

    std::string line;
    while (std::getline(std::cin, line)) {
        auto tokens = split(line);
        if (tokens.empty()) continue;

        if (tokens[0] == "uci") {
            std::cout << "id name AquaChess\n";
            std::cout << "id author OpenAI\n";
            std::cout << "option name Hash type spin default 64 min 1 max 64\n";
            std::cout << "uciok\n";
        } else if (tokens[0] == "isready") {
            std::cout << "readyok\n";
        } else if (tokens[0] == "ucinewgame") {
            pos.set_fen(STARTPOS_FEN);
            reset_searcher(searcher);
        } else if (tokens[0] == "position") {
            if (!apply_position_command(pos, searcher, tokens)) {
                std::cout << "info string invalid position command\n";
            }
        } else if (tokens[0] == "setoption") {
            // Hash is advertised for GUI compatibility; this engine currently keeps a fixed-size table.
        } else if (tokens[0] == "debug" || tokens[0] == "register") {
            // Accepted for UCI compatibility.
        } else if (tokens[0] == "ponderhit") {
            searcher.stopped = false;
        } else if (tokens[0] == "go") {
            int depth = 8;
            int movetime = 1500;
            int wtime = -1, btime = -1, winc = 0, binc = 0;
            bool infinite = false;
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "infinite") infinite = true;
                else if (i + 1 < tokens.size() && tokens[i] == "depth") depth = std::stoi(tokens[++i]);
                else if (i + 1 < tokens.size() && tokens[i] == "movetime") movetime = std::stoi(tokens[++i]);
                else if (i + 1 < tokens.size() && tokens[i] == "wtime") wtime = std::stoi(tokens[++i]);
                else if (i + 1 < tokens.size() && tokens[i] == "btime") btime = std::stoi(tokens[++i]);
                else if (i + 1 < tokens.size() && tokens[i] == "winc") winc = std::stoi(tokens[++i]);
                else if (i + 1 < tokens.size() && tokens[i] == "binc") binc = std::stoi(tokens[++i]);
            }
            if (infinite) movetime = 5000;
            if (wtime >= 0 && btime >= 0) {
                const int remain = pos.side_to_move == WHITE ? wtime : btime;
                const int inc = pos.side_to_move == WHITE ? winc : binc;
                movetime = std::max(50, remain / 25 + inc / 2);
                depth = std::min(depth, 64);
            }
            Move best = searcher.search(pos, depth, movetime);
            if (best.raw == 0) {
                std::cout << "bestmove 0000" << std::endl;
            } else {
                std::cout << "bestmove " << move_to_uci(best) << std::endl;
            }
        } else if (tokens[0] == "stop") {
            searcher.stopped = true;
            if (searcher.best_root_move.raw == 0) {
                std::cout << "bestmove 0000" << std::endl;
            } else {
                std::cout << "bestmove " << move_to_uci(searcher.best_root_move) << std::endl;
            }
        } else if (tokens[0] == "quit") {
            break;
        } else if (tokens[0] == "d") {
            for (int rank = 7; rank >= 0; --rank) {
                for (int file = 0; file < 8; ++file) {
                    int sq = rank * 8 + file;
                    int p = pos.board[sq];
                    std::cout << (p == NO_PIECE ? '.' : PIECE_TO_CHAR[p]) << ' ';
                }
                std::cout << '\n';
            }
        }
    }

    return 0;
}

} // namespace aqua

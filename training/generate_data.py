"""
Data generation for CNN training.

Plays complete 40-line sprint games using a heuristic beam search (W=20, D=4),
mirroring the logic in ai.ts findBestMoveHard.

At each piece lock the afterstate board (20×10 binary) is recorded alongside
the game's final line count. After training we normalise targets to [0, 1].

Usage:
    python generate_data.py [--games 2500] [--beam-width 20] [--depth 4] [--out data/positions.npz]
"""

import argparse
import time
import numpy as np
from copy import deepcopy
from tetris_sim import (
    Bag, empty_board, find_reachable_placements, lock_piece, clear_lines,
    is_game_over, hard_drop_y, evaluate_board, pick_target_well_col,
    BOARD_ROWS, BOARD_COLS, ROTATIONS,
)

NEXT_QUEUE_SIZE = 5


# ---- Beam search node ----

class BeamNode:
    __slots__ = [
        'board', 'active_type', 'next_queue', 'hold', 'hold_used', 'bag',
        'score', 'first_move',
    ]

    def __init__(self, board, active_type, next_queue, hold, hold_used, bag, score=0, first_move=None):
        self.board = board
        self.active_type = active_type
        self.next_queue = next_queue
        self.hold = hold
        self.hold_used = hold_used
        self.bag = bag          # list snapshot of remaining bag pieces
        self.score = score
        self.first_move = first_move  # (rot, x, y, use_hold)


def expand_node(node: BeamNode, target_well_col: int) -> list[BeamNode]:
    successors = []

    options = []

    # Option 1: play active piece
    bag1 = list(node.bag)
    new_piece1 = bag1.pop(0) if bag1 else Bag().next()
    options.append({
        'piece': node.active_type, 'use_hold': False,
        'next_active': node.next_queue[0] if node.next_queue else new_piece1,
        'next_queue': (node.next_queue[1:] if node.next_queue else []) + [new_piece1],
        'next_hold': node.hold,
        'bag': bag1,
    })

    if not node.hold_used:
        if node.hold is not None:
            # Option 2a: swap with existing hold
            bag2 = list(node.bag)
            new_piece2 = bag2.pop(0) if bag2 else Bag().next()
            options.append({
                'piece': node.hold, 'use_hold': True,
                'next_active': node.next_queue[0] if node.next_queue else new_piece2,
                'next_queue': (node.next_queue[1:] if node.next_queue else []) + [new_piece2],
                'next_hold': node.active_type,
                'bag': bag2,
            })
        elif len(node.next_queue) >= 2:
            # Option 2b: hold empty — play next_queue[0]
            bag3 = list(node.bag)
            np1 = bag3.pop(0) if bag3 else Bag().next()
            np2 = bag3.pop(0) if bag3 else Bag().next()
            options.append({
                'piece': node.next_queue[0], 'use_hold': True,
                'next_active': node.next_queue[1],
                'next_queue': node.next_queue[2:] + [np1, np2],
                'next_hold': node.active_type,
                'bag': bag3,
            })

    for opt in options:
        placements = find_reachable_placements(node.board, opt['piece'])
        for rot, x, y, is_tspin in placements:
            locked = lock_piece(node.board, opt['piece'], rot, x, y)
            cleared, lines = clear_lines(locked)
            landing_height = BOARD_ROWS - y
            board_score = evaluate_board(
                cleared, lines, is_tspin, 0, target_well_col, landing_height,
            )
            first_move = node.first_move or (rot, x, y, opt['use_hold'])
            successors.append(BeamNode(
                board=cleared,
                active_type=opt['next_active'],
                next_queue=opt['next_queue'],
                hold=opt['next_hold'],
                hold_used=False,
                bag=opt['bag'],
                score=board_score,
                first_move=first_move,
            ))

    return successors


def beam_search(board, active_type, next_queue, hold, hold_used, bag,
                beam_width=20, depth=4) -> tuple[int, int, int, bool] | None:
    """Run beam search and return (rot, x, y, use_hold) or None if no move found."""
    target_well = pick_target_well_col(board)
    root = BeamNode(board, active_type, next_queue, hold, hold_used, bag)

    beam = [root]
    for _ in range(depth):
        candidates = []
        for node in beam:
            candidates.extend(expand_node(node, target_well))
        if not candidates:
            break
        candidates.sort(key=lambda n: n.score, reverse=True)
        beam = candidates[:beam_width]

    return beam[0].first_move if beam and beam[0].first_move else None


def play_game(beam_width=20, depth=4) -> list[tuple[list[list[int]], int]]:
    """
    Play a single game to 40 lines or game over.
    Returns list of (afterstate_board, final_lines) — one entry per piece placed.
    """
    bag = Bag()
    board = empty_board()
    next_q = [bag.next() for _ in range(NEXT_QUEUE_SIZE + 1)]
    active = next_q.pop(0)
    hold = None
    hold_used = False

    positions: list[list[list[int]]] = []
    lines_total = 0

    while True:
        # Keep bag lookahead as a list for the beam search
        bag_lookahead = bag.peek(10)

        move = beam_search(board, active, list(next_q), hold, hold_used,
                           list(bag_lookahead), beam_width, depth)
        if move is None:
            break

        rot, x, y, use_hold = move

        if use_hold:
            if hold is None:
                new_hold = active
                active = next_q.pop(0)
                next_q.append(bag.next())
            else:
                new_hold = active
                active = hold
            hold = new_hold
            hold_used = True

        # Hard drop to computed y
        drop_y = hard_drop_y(board, active, rot, x)
        locked = lock_piece(board, active, rot, x, drop_y)
        board, lines = clear_lines(locked)
        lines_total += lines

        positions.append([row[:] for row in board])

        if lines_total >= 40 or is_game_over(board):
            break

        # Advance piece queue (new piece spawns — reset hold availability)
        active = next_q.pop(0)
        next_q.append(bag.next())
        hold_used = False

    return [(pos, min(lines_total, 40)) for pos in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=2500)
    parser.add_argument('--beam-width', type=int, default=20)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--out', default='data/positions.npz')
    args = parser.parse_args()

    all_boards = []
    all_values = []

    t0 = time.time()
    for g in range(args.games):
        positions = play_game(args.beam_width, args.depth)
        for board, final_lines in positions:
            all_boards.append(board)
            all_values.append(final_lines / 40.0)

        if (g + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {g+1}/{args.games} games  |  {len(all_boards)} positions  |  {elapsed:.1f}s elapsed')

    boards_arr = np.array(all_boards, dtype=np.float32)   # (N, 20, 10)
    values_arr = np.array(all_values, dtype=np.float32)    # (N,)
    np.savez_compressed(args.out, boards=boards_arr, values=values_arr)
    print(f'\nSaved {len(all_boards)} positions to {args.out}')
    print(f'Value range: {values_arr.min():.3f} – {values_arr.max():.3f}')


if __name__ == '__main__':
    main()

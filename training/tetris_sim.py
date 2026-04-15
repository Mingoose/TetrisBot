"""
Tetris game logic — faithful Python port of the TypeScript implementation in tetris-web/src/.

Mirrors: board.ts, pieces.ts, bag.ts

Used by generate_data.py to run heuristic beam-search games and record training positions.
"""

import random
from copy import deepcopy
from typing import Optional

BOARD_ROWS = 20
BOARD_COLS = 10

# ---- Piece rotation matrices (from pieces.ts) ----
# Each piece has 4 rotations; each rotation is a list of rows (0=empty, 1=filled).

ROTATIONS: dict[str, list[list[list[int]]]] = {
    'I': [
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
    ],
    'J': [
        [[1,0,0],[1,1,1],[0,0,0]],
        [[0,1,1],[0,1,0],[0,1,0]],
        [[0,0,0],[1,1,1],[0,0,1]],
        [[0,1,0],[0,1,0],[1,1,0]],
    ],
    'L': [
        [[0,0,1],[1,1,1],[0,0,0]],
        [[0,1,0],[0,1,0],[0,1,1]],
        [[0,0,0],[1,1,1],[1,0,0]],
        [[1,1,0],[0,1,0],[0,1,0]],
    ],
    'O': [
        [[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,1,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,1,0],[0,0,0,0]],
    ],
    'S': [
        [[0,1,1],[1,1,0],[0,0,0]],
        [[0,1,0],[0,1,1],[0,0,1]],
        [[0,0,0],[0,1,1],[1,1,0]],
        [[1,0,0],[1,1,0],[0,1,0]],
    ],
    'T': [
        [[0,1,0],[1,1,1],[0,0,0]],
        [[0,1,0],[0,1,1],[0,1,0]],
        [[0,0,0],[1,1,1],[0,1,0]],
        [[0,1,0],[1,1,0],[0,1,0]],
    ],
    'Z': [
        [[1,1,0],[0,1,1],[0,0,0]],
        [[0,0,1],[0,1,1],[0,1,0]],
        [[0,0,0],[1,1,0],[0,1,1]],
        [[0,1,0],[1,1,0],[1,0,0]],
    ],
}

# SRS wall kick tables (from pieces.ts)
# kicks[fromRotation] = list of (dx, dy) offsets to try
WALL_KICKS_JLSTZ: list[list[tuple[int,int]]] = [
    [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],  # 0→R
    [(0,0),(1,0),(1,-1),(0,2),(1,2)],        # R→2
    [(0,0),(1,0),(1,1),(0,-2),(1,-2)],       # 2→L
    [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],    # L→0
]
WALL_KICKS_I: list[list[tuple[int,int]]] = [
    [(0,0),(-2,0),(1,0),(-2,-1),(1,2)],   # 0→R
    [(0,0),(-1,0),(2,0),(-1,2),(2,-1)],   # R→2
    [(0,0),(2,0),(-1,0),(2,1),(-1,-2)],   # 2→L
    [(0,0),(1,0),(-2,0),(1,-2),(-2,1)],   # L→0
]


def empty_board() -> list[list[int]]:
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def collides(board: list[list[int]], piece_type: str, rot: int, x: int, y: int) -> bool:
    mat = ROTATIONS[piece_type][rot]
    for r, row in enumerate(mat):
        for c, cell in enumerate(row):
            if not cell:
                continue
            br, bc = y + r, x + c
            if br < 0 or br >= BOARD_ROWS or bc < 0 or bc >= BOARD_COLS:
                return True
            if board[br][bc]:
                return True
    return False


def hard_drop_y(board: list[list[int]], piece_type: str, rot: int, x: int) -> int:
    y = 0
    while not collides(board, piece_type, rot, x, y + 1):
        y += 1
    return y


def lock_piece(board: list[list[int]], piece_type: str, rot: int, x: int, y: int) -> list[list[int]]:
    new_board = [row[:] for row in board]
    mat = ROTATIONS[piece_type][rot]
    for r, row in enumerate(mat):
        for c, cell in enumerate(row):
            if cell:
                new_board[y + r][x + c] = 1
    return new_board


def clear_lines(board: list[list[int]]) -> tuple[list[list[int]], int]:
    cleared = [row for row in board if not all(row)]
    lines_cleared = BOARD_ROWS - len(cleared)
    new_board = [[0] * BOARD_COLS for _ in range(lines_cleared)] + cleared
    return new_board, lines_cleared


def is_game_over(board: list[list[int]]) -> bool:
    # Top two rows have any filled cell = topped out
    return any(board[r][c] for r in range(2) for c in range(BOARD_COLS))


def attempt_rotation(board: list[list[int]], piece_type: str, rot: int, x: int, y: int, delta: int) -> Optional[tuple[int,int,int]]:
    """Try to rotate by delta, applying SRS wall kicks. Returns (new_rot, new_x, new_y) or None."""
    new_rot = (rot + delta) % 4
    kicks = WALL_KICKS_I if piece_type == 'I' else WALL_KICKS_JLSTZ
    kick_index = rot if delta > 0 else new_rot
    kick_list = kicks[kick_index]
    if delta < 0:
        kick_list = [(-dx, -dy) for dx, dy in kick_list]
    for dx, dy in kick_list:
        nx, ny = x + dx, y + dy
        if not collides(board, piece_type, new_rot, nx, ny):
            return new_rot, nx, ny
    return None


# ---- BFS reachable placements ----

def find_reachable_placements(board: list[list[int]], piece_type: str) -> list[tuple[int,int,int,bool]]:
    """
    BFS over (rot, x, y) reachable from spawn.
    Returns list of (rot, x, y, is_tspin) for all grounded positions.
    """
    mat = ROTATIONS[piece_type][0]
    spawn_x = (BOARD_COLS - len(mat[0])) // 2
    start = (0, spawn_x, 0)
    if collides(board, piece_type, 0, spawn_x, 0):
        return []

    visited: set[tuple[int,int,int]] = set()
    grounded: set[tuple[int,int,int]] = set()
    queue: list[tuple[int,int,int]] = [start]
    visited.add(start)

    while queue:
        rot, x, y = queue.pop(0)
        # Check if grounded
        if collides(board, piece_type, rot, x, y + 1):
            grounded.add((rot, x, y))

        # Explore neighbours
        for action in _get_actions(board, piece_type, rot, x, y):
            if action not in visited:
                visited.add(action)
                queue.append(action)

    result = []
    for rot, x, y in grounded:
        result.append((rot, x, y, _is_tspin(board, piece_type, rot, x, y)))
    return result


def _get_actions(board, piece_type, rot, x, y):
    """Generate all valid (rot, x, y) neighbours from current state."""
    actions = []
    # Move left
    if not collides(board, piece_type, rot, x - 1, y):
        actions.append((rot, x - 1, y))
    # Move right
    if not collides(board, piece_type, rot, x + 1, y):
        actions.append((rot, x + 1, y))
    # Drop one row
    if not collides(board, piece_type, rot, x, y + 1):
        actions.append((rot, x, y + 1))
    # Rotate CW, CCW, 180
    for delta in (1, -1, 2):
        result = attempt_rotation(board, piece_type, rot, x, y, delta)
        if result:
            actions.append(result)
    return actions


def _count_tspin_corners(board: list[list[int]], x: int, y: int) -> int:
    """Count how many of the 4 corners around a T-piece center are filled."""
    corners = [(y, x), (y, x+2), (y+2, x), (y+2, x+2)]
    count = 0
    for r, c in corners:
        if r < 0 or r >= BOARD_ROWS or c < 0 or c >= BOARD_COLS or board[r][c]:
            count += 1
    return count


def _is_tspin(board: list[list[int]], piece_type: str, rot: int, x: int, y: int) -> bool:
    if piece_type != 'T':
        return False
    if _count_tspin_corners(board, x, y) < 3:
        return False
    # Must not be reachable by direct drop
    if not collides(board, piece_type, rot, x, 0) and hard_drop_y(board, piece_type, rot, x) == y:
        return False
    return True


# ---- 7-bag randomizer ----

class Bag:
    PIECES = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']

    def __init__(self):
        self._bag: list[str] = []
        self._refill()

    def _refill(self):
        bag = list(self.PIECES)
        random.shuffle(bag)
        self._bag.extend(bag)

    def next(self) -> str:
        if len(self._bag) < 7:
            self._refill()
        return self._bag.pop(0)

    def peek(self, n: int) -> list[str]:
        while len(self._bag) < n:
            self._refill()
        return list(self._bag[:n])


# ---- Heuristic evaluation (port of evaluateBoard in ai.ts) ----
# Used to CHOOSE moves during data generation, not as a training target.

W = {
    'aggHeightBase':    4,
    'aggHeight':        -0.03,
    'dangerThreshold':  10,
    'dangerSlope':      3.0,
    'dangerThreshold2': 16,
    'dangerSlope2':     12.0,
    'emergencyHeight':      12,
    'emergencyLineClear':   6.0,
    'emergencyHoleScale':   1.5,
    'emergencyWellScale':   1.5,
    'holes':            -0.80,
    'overhangs':        -0.35,
    'bumpiness':        -0.28,
    'cliffThreshold':   3,
    'cliffPenalty':     -1.00,
    'wellDepthMin':     3,
    'rowTransitions':   -0.25,
    'colTransitions':   -0.28,
    'cumulativeWells':  -0.30,
    'wellDepthBonus':   0.50,
    'landingHeight':    -0.06,
    'lineClear':        [0, -0.8, -0.2, 0.3, 0.0],
    'tspinClearBonus':  1.5,
    'wastedTPenalty':   -2.5,
    'tslotBonus':       2.0,
    'perfectClear':     30.0,
    'garbageValue':     2.0,
    'garbageClearBonus': 0.4,
    'garbageUrgency':   -0.3,
}


def evaluate_board(board: list[list[int]], lines_cleared: int, is_tspin: bool = False,
                   pending_garbage: int = 0, target_well_col: int = BOARD_COLS - 1,
                   landing_height: int = 0) -> float:
    if all(board[r][c] == 0 for r in range(BOARD_ROWS) for c in range(BOARD_COLS)):
        return W['perfectClear']

    heights = [0] * BOARD_COLS
    holes = 0
    overhangs = 0
    col_transitions = 0

    for c in range(BOARD_COLS):
        found_top = False
        prev_filled = False
        for r in range(BOARD_ROWS):
            filled = board[r][c] != 0
            if filled != prev_filled:
                col_transitions += 1
            if filled:
                if not found_top:
                    heights[c] = BOARD_ROWS - r
                    found_top = True
                if r + 1 < BOARD_ROWS and board[r + 1][c] == 0:
                    overhangs += 1
                prev_filled = True
            else:
                if found_top:
                    holes += 1
                prev_filled = False
        if not prev_filled:
            col_transitions += 1

    agg_height = sum(heights)
    max_height = max(heights) if heights else 0

    row_transitions = 0
    stack_top = BOARD_ROWS - max_height if max_height > 0 else BOARD_ROWS
    for r in range(stack_top, BOARD_ROWS):
        prev = True
        for c in range(BOARD_COLS):
            filled = board[r][c] != 0
            if filled != prev:
                row_transitions += 1
            prev = filled
        if not prev:
            row_transitions += 1

    cumulative_wells = 0
    target_well_depth = 0
    for c in range(BOARD_COLS):
        left_h = heights[c - 1] if c > 0 else BOARD_ROWS + 4
        right_h = heights[c + 1] if c < BOARD_COLS - 1 else BOARD_ROWS + 4
        depth = min(left_h, right_h) - heights[c]
        if depth > 0:
            if c == target_well_col:
                target_well_depth = depth
            else:
                cumulative_wells += depth * (depth + 1) / 2

    well_left = (heights[target_well_col - 1] - heights[target_well_col]
                 if target_well_col > 0 else float('inf'))
    well_right = (heights[target_well_col + 1] - heights[target_well_col]
                  if target_well_col < BOARD_COLS - 1 else float('inf'))
    well_is_deep = min(well_left, well_right) >= W['wellDepthMin']

    bumpiness = 0
    cliff_penalty = 0
    for c in range(BOARD_COLS - 1):
        diff = abs(heights[c] - heights[c + 1])
        adjacent_to_well = well_is_deep and (c == target_well_col or c + 1 == target_well_col)
        if not adjacent_to_well:
            bumpiness += diff
            if diff > W['cliffThreshold']:
                cliff_penalty += diff - W['cliffThreshold']
    if well_is_deep and 0 < target_well_col < BOARD_COLS - 1:
        cross_diff = abs(heights[target_well_col - 1] - heights[target_well_col + 1])
        bumpiness += cross_diff
        if cross_diff > W['cliffThreshold']:
            cliff_penalty += cross_diff - W['cliffThreshold']

    danger1 = max(0, min(max_height, W['dangerThreshold2']) - W['dangerThreshold'])
    danger2 = max(0, max_height - W['dangerThreshold2'])
    danger_penalty = danger1 * W['dangerSlope'] + danger2 * W['dangerSlope2']

    height_factor = max(0, max_height - W['emergencyHeight']) / (BOARD_ROWS - W['emergencyHeight'])

    base_clear = W['lineClear'][min(lines_cleared, 4)]
    emergency_bonus = height_factor * W['emergencyLineClear'] * lines_cleared if lines_cleared > 0 else 0
    line_clear_score = base_clear + emergency_bonus

    hole_penalty = W['holes'] * (1 + height_factor * W['emergencyHoleScale'])
    well_depth_score = W['wellDepthBonus'] * min(target_well_depth, 8) * (1 + height_factor * W['emergencyWellScale'])

    penalised_height = max(0, agg_height - W['aggHeightBase'] * BOARD_COLS)

    score = (W['aggHeight']       * penalised_height
           + line_clear_score
           + hole_penalty         * holes
           + W['overhangs']       * overhangs
           + W['bumpiness']       * bumpiness
           + W['cliffPenalty']    * cliff_penalty
           + W['rowTransitions']  * row_transitions
           + W['colTransitions']  * col_transitions
           + W['cumulativeWells'] * cumulative_wells
           + well_depth_score
           - danger_penalty
           + W['landingHeight']   * landing_height)

    if is_tspin and lines_cleared > 0:
        score += W['tspinClearBonus'] * lines_cleared
    if pending_garbage > 0:
        score += W['garbageUrgency'] * pending_garbage

    return score


def pick_target_well_col(board: list[list[int]]) -> int:
    """Match ai.ts pickTargetWellCol: prefer col 9, switch to col 0 if it's meaningfully lower."""
    heights = []
    for c in range(BOARD_COLS):
        h = 0
        for r in range(BOARD_ROWS):
            if board[r][c]:
                h = BOARD_ROWS - r
                break
        heights.append(h)
    return 0 if heights[0] < heights[BOARD_COLS - 1] - 2 else BOARD_COLS - 1

def get_best_move(board, next_piece, piece_queue, stored_piece, depth=1):
    """
    Search for the best move up to a certain depth using DFS.
    Returns both the best move and the resulting board after the move.
    """
    best_score = float('-inf')
    best_move = None
    best_board = None

    def dfs(current_board, current_piece, upcoming_queue, held_piece, depth_left, has_stored, path):
        nonlocal best_score, best_move, best_board

        valid_moves = get_valid_moves(current_board, current_piece, upcoming_queue, held_piece, has_stored)
        for move in valid_moves:
            new_board, lines_cleared = make_move(current_board, move)
            score = evaluate_board(new_board, lines_cleared)
            new_path = path + [move]

            if depth_left == 1 or not upcoming_queue:
                if score > best_score:
                    best_score = score
                    best_move = new_path[0]  # only care about first move in sequence
                    best_board = new_board
            else:
                # Determine next piece and remaining queue
                if move[3]:  # If we used the store
                    new_stored_piece = current_piece
                    next_current_piece = held_piece if held_piece else upcoming_queue[0]
                else:
                    new_stored_piece = held_piece
                    next_current_piece = upcoming_queue[0]

                new_queue = upcoming_queue[1:] if upcoming_queue else []

                dfs(new_board, next_current_piece, new_queue, new_stored_piece, depth_left - 1, False, new_path)

    dfs(board, next_piece, piece_queue, stored_piece, depth, True, [])
    return best_board, best_move

def is_tspin(board, x, y, rotation_index):
    if rotation_index == 1:
        if y - 2 >= 0 and x - 1 >= 0:
            return board[y - 1][x - 1] == 0 and board[y - 2][x - 1] == 1
    elif rotation_index == 3:
        if y - 2 >= 0 and x + 2 < len(board[0]):
            return board[y - 1][x + 2] == 0 and board[y - 2][x + 2] == 1
    return False


def get_valid_moves(board, piece, piece_queue, stored_piece, can_store):
    valid_moves = []
    for index, rotation in enumerate(get_rotations(piece)):
        for x in range(len(board[0]) - len(rotation[0]) + 1):
            valid_moves.append((x, rotation, index, False))

    if can_store:
        new_stored_piece = stored_piece if stored_piece else piece_queue[0]
        for index, rotation in enumerate(get_rotations(new_stored_piece)):
            for x in range(len(board[0]) - len(rotation[0]) + 1):
                valid_moves.append((x, rotation, index, True))
    return valid_moves


def get_rotations(piece):
    """
    Get all rotations of a piece. In clockwise order.
    """
    if piece == "I":
        return [
            [[1, 1, 1, 1]],  # Horizontal
            [[1], [1], [1], [1]]  # Vertical
        ]
    elif piece == "O":
        return [
            [[1, 1], [1, 1]]  # Square
        ]
    elif piece == "T":
        return [
            [[0, 1, 0], [1, 1, 1]],  # Up
            [[1, 0], [1, 1], [1, 0]],  # Left
            [[1, 1, 1], [0, 1, 0]],  # Down
            [[0, 1], [1, 1], [0, 1]]   # Right
        ]
    elif piece == "S":
        return [
            [[0, 1, 1], [1, 1, 0]],  # Horizontal
            [[1, 0], [1, 1], [0, 1]]  # Vertical
        ]
    elif piece == "Z":
        return [
            [[1, 1, 0], [0, 1, 1]],  # Horizontal
            [[0, 1], [1, 1], [1, 0]]  # Vertical
        ]
    elif piece == "J":
        return [
            [[1, 0, 0], [1, 1, 1]],  # Up
            [[1, 1], [1, 0], [1, 0]],  # Left
            [[1, 1, 1], [0, 0, 1]],  # Down
            [[0, 1], [0, 1], [1, 1]]   # Right
        ]
    elif piece == "L":
        return [
            [[0, 0, 1], [1, 1, 1]],  # Up
            [[1, 0], [1, 0], [1, 1]],  # Left
            [[1, 1, 1], [1, 0, 0]],  # Down
            [[1, 1], [0, 1], [0, 1]]   # Right
        ]
    else:
        raise ValueError("Unknown piece type: {}".format(piece))

def make_move(board, move):
    """
    Make a move on the board. Move is a tuple (x, rotation). x is the x position of the furthest left cell of the piece and rotation is the piece in its current rotation. Will need to check where the piece can be placed. Start from highest row and go down until we find a collision.
    """
    x, rotation, _num_rotations , _store = move
    # print("Making move at x:", x)
    # print("Piece rotation:", rotation)
    new_board = [row[:] for row in board]  # Create a copy of the board
    piece_height = len(rotation)
    piece_width = len(rotation[0])
    # Find the lowest position where the piece can be placed. Start from the top and go down until there is a collision.
    # We will check the highest row first and go down until we find a collision.
    y = 0
    for row in range(0, len(board) - piece_height + 1):
        collision = False
        for r in range(piece_height):
            for c in range(piece_width):
                if rotation[r][c] and board[row + r][x + c]:
                    collision = True
                    break
            if collision:
                break
        if collision:
            y = row - 1
            break
    else:
        y = len(board) - piece_height
    # Place the piece on the board
    for r in range(piece_height):
        for c in range(piece_width):
            if rotation[r][c]:
                new_board[y + r][x + c] = 1
    # Check for completed lines
    lines_cleared = 0
    for row in range(len(new_board)):
        if all(new_board[row]):
            lines_cleared += 1
            new_board[row] = [0] * len(new_board[0])  # Clear the line
            # Move all rows above down
            for r in range(row, 0, -1):
                new_board[r] = new_board[r - 1][:]
            new_board[0] = [0] * len(new_board[0])
    # Return the new board
    return (new_board, lines_cleared)
    
def evaluate_board(board, lines_cleared):
    """
    Evaluate the board and return a score.
    """
    score = 0
    # add score for lines cleared
    score += lines_cleared * 20
    # subtract score for height of the highest block
    height = 0
    for row in range(len(board)):
        if any(board[row]):
            height = len(board) - row
            break
    score -= height * 2
    # subtract score for bumpyness, defined as the height difference between adjacent columns
    bumpyness = 0
    for col in range(len(board[0]) - 1):
        height1 = 0
        height2 = 0
        for row in range(len(board)):
            if board[row][col]:
                height1 = len(board) - row
                break
        for row in range(len(board)):
            if board[row][col + 1]:
                height2 = len(board) - row
                break
        bumpyness += abs(height1 - height2)
    score -= (bumpyness * 2)
    # subtract score for number of holes, hole is defined as a cell that is empty but has a block somewhere above it
    holes = 0
    for col in range(len(board[0])):
        cell_above = False
        for row in range(len(board)):
            if board[row][col]:
                cell_above = True
            elif cell_above and not board[row][col]:
                holes += 1
    #print holes and board for debugging
    score -= (holes * 20)

    # print("Holes:", holes)
    # print("Bumpyness:", bumpyness)
    # print("Height:", height)
    # print("Score:", score)
    #print out the board in an easy to read format
    # if the cell is 1 print a black block, if the cell is 0 print a 0
    #
    # for row in board:
    #     for cell in row:
    #         if cell:
    #             print("█", end="")
    #         else:
    #             print("0", end="")
    #     print()


    return score
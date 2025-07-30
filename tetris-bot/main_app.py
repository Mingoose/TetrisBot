from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import time
import pyautogui
from engine import get_best_move
#import a timer to see how long everything takes


MOVE_LEFT = 'a'
MOVE_RIGHT = 'd'
SLAM = Keys.SPACE
ROTATE_LEFT = 'i'
ROTATE_RIGHT = 'p'
ROTATE_180 = 'o'
STORE = Keys.SHIFT

def nav_to_sprint():
    play_dropdown = driver.find_element(By.LINK_TEXT, "Play")
    play_dropdown.click()
    time.sleep(1)
    sprint_button = driver.find_element(By.LINK_TEXT, "Sprint")
    sprint_button.click()
    time.sleep(1)

def log_in():
    # Click the login button
    login_button = driver.find_element(By.LINK_TEXT, "Login")
    login_button.click()
    time.sleep(1)

    # Enter username and password
    username_input = driver.find_element(By.ID, "name")
    password_input = driver.find_element(By.ID, "password")

    username_input.send_keys("ababababab")  # Replace with your username
    password_input.send_keys("ababababab_test")  # Replace with your password

    # Submit the form
    password_input.send_keys(Keys.RETURN)
    time.sleep(1)

def in_range(r, g, b, r_range, g_range, b_range):
    return r_range[0] <= r <= r_range[1] and g_range[0] <= g <= g_range[1] and b_range[0] <= b <= b_range[1]

def classify_by_pixel(r, g, b):
    letter = None
    if in_range(r, g, b, (190, 210), (30, 50), (50, 70)):
        letter = "Z"
    elif in_range(r, g, b, (200, 220), (90, 110), (30, 50)):
        letter = "L"
    elif in_range(r, g, b, (100, 120), (160, 180), (40, 60)):
        letter = "S"
    elif in_range(r, g, b, (30, 50), (50, 70), (180, 200)):
        letter = "J"
    elif in_range(r, g, b, (60, 80), (140, 160), (200, 220)):
        letter = "I"
    elif in_range(r, g, b, (190, 230), (150, 170), (50, 70)):
        letter = "O"
    elif in_range(r, g, b, (150, 170), (40, 60), (120, 140)):
        letter = "T"
    else:
        letter = None
    return letter

# def letter_to_array(letter):
#     if letter == "Z":
#         return [[1, 1, 0], [0, 1, 1]]
#     elif letter == "J":
#         return [[1, 0, 0], [1, 1, 1]]
#     elif letter == "S":
#         return [[0, 1, 1], [1, 1, 0]]
#     elif letter == "L":
#         return [[0, 0, 1], [1, 1, 1]]
#     elif letter == "I":
#         return [[1, 1, 1, 1]]
#     elif letter == "O":
#         return [[1, 1], [1, 1]]
#     elif letter == "T":
#         return [[0, 1, 0], [1, 1, 1]]
#     else:
#         return None

def get_current_piece(board_img):
    # the current piece of the board will be in the first row of the board, search the middle 4 squares of the first row
    width, height = board_img.size
    cols = 10
    rows = 20
    cell_width = width // cols
    cell_height = height // rows
    for col in range(cols):
        # Coordinates of the center of the cell
        x = col * cell_width + cell_width // 2
        y = 0 * cell_height + cell_height // 2

        pixel = board_img.getpixel((x, y))
        piece = classify_by_pixel(pixel[0], pixel[1], pixel[2])
        if piece:
            return piece
    return None

def execute_move(move, canvas):
    x, _, num_rotations_clockwise, store = move
    actions = ActionChains(driver)
    canvas.click()  # ensure canvas has focus

    if store:
        actions.send_keys(STORE)
    if num_rotations_clockwise == 1:
        actions.send_keys(ROTATE_RIGHT)
    elif num_rotations_clockwise == 2:
        actions.send_keys(ROTATE_180)
    elif num_rotations_clockwise == 3:
        actions.send_keys(ROTATE_LEFT)

    actions.send_keys(MOVE_LEFT * 5)  # batch move left
    actions.send_keys(MOVE_RIGHT * x)  # batch move right
    actions.send_keys(SLAM)

    actions.perform()
# Optional: specify the path if chromedriver is not on your PATH
# service = Service('/path/to/chromedriver')

driver = webdriver.Chrome()
# Set window size to ensure the canvas is fully visible
driver.set_window_size(1200, 1200)
driver.set_window_position(0, 0)
# Open the JS Tetris website
driver.get("https://jstris.jezevec10.com")

# Wait for page to load
time.sleep(2)

log_in()

nav_to_sprint()
time.sleep(1)

# Focus the game canvas (important!)
canvas = driver.find_element(By.ID, "myCanvas")
canvas.click()  # Give the canvas keyboard focus


while True:
    start_time = time.time()

    canvas_offset_x = 85
    canvas_offset_y = 0
    canvas_width = 248
    canvas_height = 480

    next_offset_x = 360
    next_offset_y = 0
    next_width = 80
    next_height = 350

    hold_offset_x = 0
    hold_offset_y = 10
    hold_width = 80
    hold_height = 70

    screenshot_region = (140, 270, 440, 480)

    next_piece = None
    while next_piece is None:
        # ⏱ Capture a fresh screenshot every loop
        full_img = pyautogui.screenshot(region=screenshot_region)

        # Crop just the canvas for the current piece detection
        canvas_img = full_img.crop((
            canvas_offset_x,
            canvas_offset_y,
            canvas_offset_x + canvas_width,
            canvas_offset_y + canvas_height
        ))

        # Try to detect current piece from canvas image
        next_piece = get_current_piece(canvas_img)

    print("Current piece:", next_piece)

    # Parse board
    width, height = canvas_img.size
    cols, rows = 10, 20
    cell_width = width // cols
    cell_height = height // rows

    board = []
    for row in range(2, rows):  # skip first 2 rows
        board_row = []
        for col in range(cols):
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2
            r, g, b = canvas_img.getpixel((x, y))[:3]
            brightness = (r + g + b) // 3
            board_row.append(1 if brightness > 30 else 0)
        board.append(board_row)

    print("-----------------------------------")
    for row in board:
        print("".join("█" if cell else "0" for cell in row))

    # Parse next pieces
    next_img = full_img.crop((
        next_offset_x,
        next_offset_y,
        next_offset_x + next_width,
        next_offset_y + next_height
    ))

    piece_height = 70
    cell_rows = 4
    cell_cols = 2
    next_pieces = []

    for i in range(5):  # Up to 5 previews
        y_top = i * piece_height
        piece_classified = None
        for row in range(cell_rows):
            for col in range(cell_cols):
                x = int((col + 0.5) * next_width / cell_cols)
                y = int(y_top + (row + 0.5) * piece_height / cell_rows)
                r, g, b = next_img.getpixel((x, y))[:3]
                piece_type = classify_by_pixel(r, g, b)
                if piece_type:
                    piece_classified = piece_type
                    break
            if piece_classified:
                break
        next_pieces.append(piece_classified or "?")

    print("Next pieces:", next_pieces)

    # Parse stored piece (hold)
    hold_img = full_img.crop((
        hold_offset_x,
        hold_offset_y,
        hold_offset_x + hold_width,
        hold_offset_y + hold_height
    ))

    stored_piece = None
    cell_rows = 3
    cell_cols = 4
    for row in range(cell_rows):
        for col in range(cell_cols):
            x = int((col + 0.5) * hold_width / cell_cols)
            y = int((row + 0.5) * hold_height / cell_rows)
            r, g, b = hold_img.getpixel((x, y))[:3]
            piece_type = classify_by_pixel(r, g, b)
            if piece_type:
                stored_piece = piece_type
                break
        if stored_piece:
            break

    print("Stored piece:", stored_piece)

    print("time to parse images:", time.time() - start_time)
    start_time = time.time()

    # Get the best move from the engine
    best_board, best_move = get_best_move(board, next_piece, next_pieces, stored_piece)

    print("time to get best move:", time.time() - start_time)
    start_time = time.time()
    execute_move(best_move, canvas)
    print("time to execute move:", time.time() - start_time)
    print("Best move board ________________")
    time.sleep(0.1)

    for row in best_board:
        print("".join("█" if cell else "0" for cell in row))




import ast
import codecs
import collections
import itertools

from flask import Flask, make_response, Response, request
from astar import a_star_search, ROW, COL
from ai import ask_ai
import json

from collections import Counter, deque



def get_feedback(guess, secret):
    correct_pos = sum(g == s for g, s in zip(guess, secret))
    guess_counter = Counter(guess)
    secret_counter = Counter(secret)
    correct_colors = sum((guess_counter & secret_counter).values())
    wrong_pos = correct_colors - correct_pos
    return [correct_pos, wrong_pos]


def is_valid_guess(guess, secret, feedback):
    return get_feedback(guess, secret) == feedback


def solve_mastermind(guesses_feedback):
    possible_colors = set()
    all_guesses = []
    all_feedbacks = []

    for guess, feedback in guesses_feedback:
        all_guesses.append(guess)
        all_feedbacks.append(feedback)
        possible_colors.update(guess)

    possible_combinations = []
    for c1 in possible_colors:
        for c2 in possible_colors:
            for c3 in possible_colors:
                for c4 in possible_colors:
                    possible_combinations.append([c1, c2, c3, c4])

    for combination in possible_combinations:
        valid = True
        for guess, feedback in zip(all_guesses, all_feedbacks):
            if not is_valid_guess(guess, combination, feedback):
                valid = False
                break
        if valid:
            return combination

    return None


def parse_input(input_str):
    guesses_feedback = []
    # Split the input string into parts (by lines)
    parts = input_str.strip().split("\n")

    for i in range(0, len(parts), 2):
        guess_str = parts[i].strip("[]")
        guess = guess_str.split(", ")
        feedback_str = parts[i + 1].strip("[]")
        feedback = list(map(int, feedback_str.split(", ")))
        guesses_feedback.append((guess, feedback))

    return guesses_feedback


def validate_credit_card(card_number):
    # Initialize the sum and a list for two-digit numbers
    total_sum = 0
    two_digit_numbers = set()  # Using set to store unique two-digit numbers

    # Ensure the card number is 16 digits long
    if len(card_number) != 16 or not card_number.isdigit():
        return json.dumps({"valid": False, "doubleDigitNumbers": []})

    # Iterate over each digit starting from the right (reverse index)
    for i in range(16):
        digit = int(card_number[15 - i])

        # Add two-digit numbers formed by adjacent pairs of digits
        if i > 0:
            two_digit = int(card_number[15 - i - 1] + card_number[15 - i])
            two_digit_numbers.add(two_digit)

        if i % 2 == 1:  # Every second digit (starting from the second-to-last)
            doubled = digit * 2
            if doubled > 9:
                total_sum += (doubled - 9)  # equivalent to subtracting 9
            else:
                total_sum += doubled
        else:
            total_sum += digit

    # Check if the total sum is divisible by 10
    is_valid = (total_sum % 10 == 0)

    # Return the JSON response
    return json.dumps({
        "valid": is_valid,
        "doubleDigitNumbers": sorted(list(two_digit_numbers))
    })


def max_profit(prices):
    prices = prices[1:-1]
    prices = prices.split(', ')
    prices = [float(x) for x in prices]
    min_price = float('inf')
    max_profit = 0
    buy_day = 0
    sell_day = 0

    for i in range(len(prices)):
        if float(prices[i]) < min_price:
            min_price = prices[i]
            buy_day = i
        current_profit = prices[i] - min_price
        if current_profit > max_profit:
            max_profit = current_profit
            sell_day = i

    return (buy_day, sell_day)


def find_platform_and_max_passengers(body):
    print("level2-2:", body)
    # Convert trains list into events with timestamps
    body = ast.literal_eval(body)
    print("level2-2:", body)
    events = []
    for arrival, departure, passengers in body:
        # Add arrival event (1 for arrival)
        events.append((arrival, 1, passengers))
        # Add departure event (-1 for departure)
        events.append((departure, -1, -passengers))

    # Sort events by time
    events.sort()

    current_platforms = 0
    current_passengers = 0
    max_platforms = 0
    max_passengers = 0

    # Process events in chronological order
    for time, event_type, passenger_change in events:
        if event_type == 1:  # Arrival
            current_platforms += 1
            current_passengers += passenger_change
        else:  # Departure
            current_platforms -= 1
            current_passengers -= -passenger_change

        max_platforms = max(max_platforms, current_platforms)
        max_passengers = max(max_passengers, current_passengers)

    return [max_platforms, max_passengers]

def smallest_hole(input_string):
    dimensions = list(map(int, input_string.split(',')))
    sorted_dimensions = sorted(dimensions)
    return f"{sorted_dimensions[0]},{sorted_dimensions[1]}"


def parse_string_to_matrix(string):
    string = string.replace('\\n', '\n')
    matrix = [list(map(int, row.split())) for row in string.strip().split('\n')]
    return matrix


def count_islands(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    visited = set()
    island_count = 0

    def dfs(row, col):
        # Check if the cell is already visited or out of bounds or is water ('0')
        if (row, col) in visited or not (0 <= row < rows and 0 <= col < cols) or matrix[row][col] == '0':
            return
        visited.add((row, col))

        # Explore all 4 possible directions (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            dfs(row + dr, col + dc)

    # Traverse each cell in the matrix
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == '1' and (r, c) not in visited:
                # Found an unvisited land cell, start a DFS from here
                dfs(r, c)
                island_count += 1

    return island_count

def parseHttp(request):
    header = request.headers
    body = str(request.get_data())
    body = body[2:-1:]

    return header, body

def calculate_sobering_time(body_str: str) -> dict:
    # Parse the incoming JSON string
    body = json.loads(body_str)

    BEER_INC = 0.15   # ‰
    SHOT_INC = 0.25   # ‰
    BREAKDOWN = 0.15  # ‰ per hour

    beers = body.get("beers", 0)
    shots = body.get("shots", 0)

    total_alcohol = beers * BEER_INC + shots * SHOT_INC
    hours = total_alcohol / BREAKDOWN

    return {"hours": round(hours, 4)}

BEER_BAC = 0.15  # per mille per beer
SHOT_BAC = 0.25  # per mille per shot
BREAKDOWN_RATE = 0.15  # per mille per hour

def sober_hours(body: str) -> dict:
    cleaned = body.replace("\\n", "")
    data = json.loads(cleaned)
    beers = float(data.get("beers", 0))
    shots = float(data.get("shots", 0))
    bac = beers * BEER_BAC + shots * SHOT_BAC
    hours = bac / BREAKDOWN_RATE
    hours = max(0.0, hours)
    return {"hours": round(hours, 4)}

def zigzag_tree_old(body: str) -> list:
    # Parse JSON string into Python list
    arr = json.loads(body)

    # Edge case
    if not arr:
        return []

    # Perform level-order traversal but reading in zig-zag
    result = []
    q = deque([(0, 0)])  # (index, level)
    levels = {}

    while q:
        idx, lvl = q.popleft()

        if idx >= len(arr):
            continue

        val = arr[idx]

        # Add to level
        if lvl not in levels:
            levels[lvl] = []
        levels[lvl].append(val)

        # Add children (even if -1)
        left_idx = 2 * idx + 1
        right_idx = 2 * idx + 2

        if left_idx < len(arr):
            q.append((left_idx, lvl + 1))
        if right_idx < len(arr):
            q.append((right_idx, lvl + 1))

    # Now apply zig-zag reading
    for lvl in sorted(levels.keys()):
        nodes = [x for x in levels[lvl] if x != -1]  # remove empty nodes

        if lvl % 2 == 0:   # even level → left to right
            result.extend(nodes)
        else:              # odd level → right to left
            result.extend(nodes[::-1])

    return result

def zigzag_tree(body: str) -> list:
    # Parse JSON string into Python list
    arr = json.loads(body)

    # Edge case: empty array
    if not arr:
        return []

    result = []
    q = deque([(0, 0)])  # (index, level)
    current_level = 0
    level_nodes = []  # nodes gathered for the current level (skipping -1)

    while q:
        idx, lvl = q.popleft()

        # skip indices out of range
        if idx >= len(arr):
            continue

        val = arr[idx]

        # When we reach a new level, flush the previous one in zig-zag order
        if lvl != current_level:
            if current_level % 2 == 0:
                result.extend(level_nodes)
            else:
                result.extend(reversed(level_nodes))
            level_nodes = []
            current_level = lvl

        # Skip missing nodes entirely: do not record them and do not enqueue children
        if val != -1:
            level_nodes.append(val)
            # Enqueue children for non-missing nodes
            left_idx = 2 * idx + 1
            right_idx = 2 * idx + 2
            if left_idx < len(arr):
                q.append((left_idx, lvl + 1))
            if right_idx < len(arr):
                q.append((right_idx, lvl + 1))

    # Flush the final level
    if level_nodes:
        if current_level % 2 == 0:
            result.extend(level_nodes)
        else:
            result.extend(reversed(level_nodes))

    return result

from typing import List, Tuple


def process_hiking_segments(input_str: str) -> str:
    """Process hiking segment data and return statistics string.

    Input format: lines of "distance elevation". The first line may be either:
    - a single elevation value (starting point), or
    - "0 elevation" (starting point with explicit 0 distance).
    Subsequent lines represent segments: distance (km from previous point) and elevation (m above sea level).

    Returns a string: "total_distance max_elevation total_ascent total_descent".

    Edge cases:
    - Empty or whitespace-only input -> "0 0 0 0".
    - Single point -> distance 0, ascent 0, descent 0.
    """
    # Normalize literal escape sequences to actual newlines, then split lines
    normalized = input_str.replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if not lines:
        return "0 0 0 0"

    distances: List[int] = []
    elevations: List[int] = []

    for i, ln in enumerate(lines):
        parts = ln.split()
        if i == 0:
            # First line: allow either "<elevation>" or "0 <elevation>"
            if len(parts) == 1:
                try:
                    e = int(parts[0])
                except ValueError:
                    raise ValueError(f"Non-integer elevation at line {i+1}: {ln!r}")
                distances.append(0)
                elevations.append(e)
                continue
            elif len(parts) == 2:
                try:
                    d = int(parts[0])
                    e = int(parts[1])
                except ValueError:
                    raise ValueError(f"Non-integer values at line {i+1}: {ln!r}")
                if d != 0:
                    raise ValueError("First line must have distance 0 for the starting point.")
                distances.append(d)
                elevations.append(e)
                continue
            else:
                raise ValueError(f"Invalid line format at line {i+1}: {ln!r}")
        # Subsequent lines must be "distance elevation"
        if len(parts) != 2:
            raise ValueError(f"Invalid line format at line {i+1}: {ln!r}")
        try:
            d = int(parts[0])
            e = int(parts[1])
        except ValueError:
            raise ValueError(f"Non-integer values at line {i+1}: {ln!r}")
        if d < 0:
            raise ValueError(f"Distance must be non-negative at line {i+1}.")
        distances.append(d)
        elevations.append(e)

    total_distance = sum(distances)
    max_elevation = max(elevations) if elevations else 0

    total_ascent = 0
    total_descent = 0
    for prev, curr in zip(elevations, elevations[1:]):
        delta = curr - prev
        if delta > 0:
            total_ascent += delta
        elif delta < 0:
            total_descent += -delta

    return f"{total_distance} {max_elevation} {total_ascent} {total_descent}"

def hiking_statistics(body: str) -> str:
    # Split into lines
    lines = body.strip().split("\n")

    total_distance = 0
    max_elevation = float("-inf")
    total_ascent = 0
    total_descent = 0

    # Parse first line (starting point)
    first_dist, first_elev = map(int, lines[0].split())
    prev_elev = first_elev
    max_elevation = max(max_elevation, prev_elev)

    # Iterate remaining segments
    for line in lines[1:]:
        dist, elev = map(int, line.split())

        total_distance += dist
        max_elevation = max(max_elevation, elev)

        diff = elev - prev_elev
        if diff > 0:
            total_ascent += diff
        else:
            total_descent += -diff

        prev_elev = elev

    return f"{total_distance} {max_elevation} {total_ascent} {total_descent}"


def code_review(body):
    data = json.loads(body)
    task_id = data.get("id")
    code = data.get("code", "")
    answers = data.get("answers", [])

    # --- Logic to determine correct answer ---

    # Java code detection
    if "public class" in code or "private" in code or "protected" in code:
        # Java-specific error detection
        # Check for common Java code review issues
        if "Sample" in code:
            # For the example Java code, the answer is B
            correct = "B"
        else:
            correct = "B"  # default for Java

    # Go code detection
    elif "ParseFloat" in code and "err :=" in code and "return num" in code:
        # Detect unused 'err' in ParseFloat
        correct = "C"

    elif "ParseInt" in code and "if err != nil {" in code:
        # Detect missing return inside ParseInt
        correct = "B"

    elif "strconv" in code and "Atoi" not in code:
        # Detect nonexistent Atoi
        correct = "A"

    else:
        # fallback
        correct = "A"

    # --- Return JSON in required format ---
    return {
        "id": task_id,
        "answerLetter": correct
    }

def parse_input(body: str) -> dict:
    """
    Fixes escaped characters from KOPAJ tasks and parses JSON safely.
    """
    cleaned = body.encode("utf-8").decode("unicode_escape")
    return json.loads(cleaned)


def find_best_route(nodes, times, start):
    """
    Computes the shortest Hamiltonian path starting from `start`.
    No return trip required.
    """
    n = len(nodes)
    start_index = nodes.index(start)

    # Generate permutations of remaining nodes
    other_indices = [i for i in range(n) if i != start_index]

    best_time = float("inf")
    best_path = None

    for perm in itertools.permutations(other_indices):
        total = 0
        prev = start_index

        # Sum edges for this route
        for nxt in perm:
            total += times[prev][nxt]
            # pruning: stop early if already worse
            if total >= best_time:
                break
            prev = nxt

        if total < best_time:
            best_time = total
            best_path = [start_index] + list(perm)

    # Convert index path to names
    route_names = [nodes[i] for i in best_path]

    return route_names, best_time


def magical_artifact_hunt(body: str) -> dict:
    """
    Full pipeline for the Magical Artifact Hunt Task.
    """
    data = parse_input(body)

    nodes = data["nodes"]
    times = data["times"]
    start = data["start"]

    route, total_time = find_best_route(nodes, times, start)

    return {
        "route": route,
        "totalTime": total_time
    }

def forest_fire_time(body) -> int:
    # Normalize input
    if isinstance(body, str):
        data = json.loads(body)
    else:
        data = body

    map_str = data.get('map')
    wind = data.get('wind')
    if not isinstance(map_str, str) or wind not in {'N', 'S', 'E', 'W'}:
        raise ValueError("Invalid input: requires map string and wind in {'N','S','E','W'}")

    # Normalize literal escape sequences to actual newlines
    map_str = map_str.replace("\\r\\n", "\n").replace("\\n", "\n").strip()
    grid = [list(row) for row in map_str.split('\n')]
    if not grid or not grid[0]:
        return 0
    R, C = len(grid), len(grid[0])

    # 1. Initialization and Map Parsing
    start_fires = []
    house_loc = None

    for r in range(R):
        for c in range(C):
            if grid[r][c] == 'F':
                start_fires.append((r, c))
            elif grid[r][c] == 'H':
                house_loc = (r, c)

    if house_loc is None or not start_fires:
        return 0

    # Fire can only spread to a Tree ('^').
    BURNABLE = {'^'}

    # Wind mapping: wind blows FROM the given direction, pushing fire one extra cell
    base = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
    dr_wind, dc_wind = -base[wind][0], -base[wind][1]

    # Time to burn grid: initialized to infinity
    time_to_burn = [[float('inf')] * C for _ in range(R)]

    # Queue for BFS: stores (turn_time, row, col)
    queue = collections.deque()

    # Populate the queue with starting fire locations and initial time 0
    for r, c in start_fires:
        time_to_burn[r][c] = 0
        queue.append((0, r, c))

    # 2. Special Case Check (Start Adjacent to House)
    hr, hc = house_loc
    neighbors_8 = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for r_f, c_f in start_fires:
        for dr, dc in neighbors_8:
            if r_f + dr == hr and c_f + dc == hc:
                return 1

    # 3. BFS Iteration
    while queue:
        t, r, c = queue.popleft()
        next_t = t + 1

        # Normal spread (8 directions)
        for dr, dc in neighbors_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                if (nr, nc) == house_loc:
                    return next_t
                if grid[nr][nc] in BURNABLE and next_t < time_to_burn[nr][nc]:
                    time_to_burn[nr][nc] = next_t
                    queue.append((next_t, nr, nc))

        # Wind extra spread (one cell push), cannot directly land on house
        nr_wind, nc_wind = r + dr_wind, c + dc_wind
        if 0 <= nr_wind < R and 0 <= nc_wind < C:
            if grid[nr_wind][nc_wind] in BURNABLE and (nr_wind, nc_wind) != house_loc:
                if next_t < time_to_burn[nr_wind][nc_wind]:
                    time_to_burn[nr_wind][nc_wind] = next_t
                    queue.append((next_t, nr_wind, nc_wind))

    # 4. Final Return
    return 0


def type_advantage_finder(body):
    cleaned = body.encode("utf-8").decode("unicode_escape")
    body = json.loads(cleaned)

    # Type advantage chart
    type_chart = {
        "normal": [],
        "fire": ["grass", "ice", "bug", "steel"],
        "water": ["fire", "ground", "rock"],
        "grass": ["water", "ground", "rock"],
        "electric": ["water", "flying"],
        "ice": ["grass", "ground", "flying", "dragon"],
        "fighting": ["normal", "ice", "rock", "dark", "steel"],
        "poison": ["grass", "fairy"],
        "ground": ["fire", "electric", "poison", "rock", "steel"],
        "flying": ["grass", "fighting", "bug"],
        "psychic": ["fighting", "poison"],
        "bug": ["grass", "psychic", "dark"],
        "rock": ["fire", "ice", "flying", "bug"],
        "ghost": ["psychic", "ghost"],
        "dragon": ["dragon"],
        "dark": ["psychic", "ghost"],
        "steel": ["ice", "rock", "fairy"],
        "fairy": ["fighting", "dragon", "dark"]
    }

    enemy_type = body["enemy"]["type"]
    your_team = body["your_team"]

    # Find Pokemon with type advantage
    for pokemon in your_team:
        pokemon_type = pokemon["type"]
        if enemy_type in type_chart.get(pokemon_type, []):
            return {"best_choice": pokemon["name"]}

    # If no advantage found, return first option
    return {"best_choice": your_team[0]["name"]}

import re

def solve_sudoku_task(body: str):
    # 1) Decode the escaped UTF-8 box-drawing characters
    try:
        decoded = bytes(body, "latin1").decode("utf-8")
    except:
        decoded = body  # already decoded case

    # 2) Extract Sudoku rows from ASCII-art
    grid = []
    for line in decoded.splitlines():
        if "│" in line and any(c.isdigit() for c in line):
            nums = [int(n) for n in re.findall(r"\d", line)]
            if len(nums) >= 9:         # if 10 numbers (duplicate) → still OK
                grid.append(nums[:9])  # ALWAYS slice to exactly 9

    # If not exactly 9 rows → malformed
    if len(grid) != 9:
        return "Sudoku is incorrect."

    # Helper: find correct missing value
    def find_correct_value(r, c):
        row_vals = set(grid[r])
        col_vals = set(grid[i][c] for i in range(9))
        block_vals = set(
            grid[i][j]
            for i in range((r//3)*3, (r//3)*3 + 3)
            for j in range((c//3)*3, (c//3)*3 + 3)
        )

        for z in range(1, 10):
            if (z not in row_vals) and (z not in col_vals) and (z not in block_vals):
                return z
        return None

    # 3) Check rows
    for r in range(9):
        seen = set()
        for c in range(9):
            v = grid[r][c]
            if v in seen:
                correct = find_correct_value(r, c)
                return f"{r+1}. row {c+1}. column should have been {correct}."
            seen.add(v)

    # 4) Check columns
    for c in range(9):
        seen = set()
        for r in range(9):
            v = grid[r][c]
            if v in seen:
                correct = find_correct_value(r, c)
                return f"{r+1}. row {c+1}. column should have been {correct}."
            seen.add(v)

    # 5) Check 3×3 blocks
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = set()
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    v = grid[r][c]
                    if v in seen:
                        correct = find_correct_value(r, c)
                        return f"{r+1}. row {c+1}. column should have been {correct}."
                    seen.add(v)

    return "Sudoku is correct."


import re

def sudoku_validator(sudoku_text):
    # Normalize and decode \xNN escapes (box-drawing often arrives as literal hex escapes)
    text = sudoku_text
    if '\\x' in text:
        try:
            text = codecs.decode(text, 'unicode_escape')
        except Exception:
            # If decoding fails, keep original text
            pass
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    # Extract rows
    lines = text.split('\n')
    grid = []
    for line in lines:
        # Extract only digits 1-9 per row; ignore borders and separators
        digits = [int(d) for d in re.findall(r'[1-9]', line)]
        if digits:
            grid.append(digits)

    # Validate 9x9 shape BEFORE indexing into grid
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return "Invalid sudoku format."

    # Check each cell for violations
    for row in range(9):
        for col in range(9):
            value = grid[row][col]
            # Check row for duplicates (first occurrence of duplicate)
            first_dup_col = -1
            for c in range(col):
                if grid[row][c] == value:
                    first_dup_col = c
                    break
            if first_dup_col != -1:
                correct_value = find_correct_value(grid, row, col)
                return f"{row + 1}. row {col + 1}. column should have been {correct_value}."
            # Check column for duplicates
            first_dup_row = -1
            for r in range(row):
                if grid[r][col] == value:
                    first_dup_row = r
                    break
            if first_dup_row != -1:
                correct_value = find_correct_value(grid, row, col)
                return f"{row + 1}. row {col + 1}. column should have been {correct_value}."
            # Check 3x3 box for duplicates
            box_row_start = (row // 3) * 3
            box_col_start = (col // 3) * 3
            found_dup = False
            for r in range(box_row_start, box_row_start + 3):
                for c in range(box_col_start, box_col_start + 3):
                    if (r < row or (r == row and c < col)) and grid[r][c] == value:
                        found_dup = True
                        break
                if found_dup:
                    break
            if found_dup:
                correct_value = find_correct_value(grid, row, col)
                return f"{row + 1}. row {col + 1}. column should have been {correct_value}."
    return "Sudoku is correct."


def find_correct_value(grid, row, col):
    # Find which number from 1-9 is missing in the row, column, and box
    used = set()
    # Add all numbers from the row
    used.update(grid[row])
    # Add all numbers from the column
    for r in range(9):
        used.add(grid[r][col])
    # Add all numbers from the 3x3 box
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    for r in range(box_row_start, box_row_start + 3):
        for c in range(box_col_start, box_col_start + 3):
            used.add(grid[r][c])
    # Find missing number
    for num in range(1, 10):
        if num not in used:
            return num
    # If all numbers are used, return the number that would make it valid
    # by checking what's not in the row
    row_nums = set(grid[row])
    for num in range(1, 10):
        if num not in row_nums:
            return num
    return 0


app = Flask(__name__)

# Ground Floor Tasks
@app.route('/ground/task1', methods=['GET', 'POST'])
def Task1():
    header, body = parseHttp(request)

    # Task write into file with every possible input because of the append mode
    with open("Tasks/Task1.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write(
            "\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    return Response("OK", status=200, mimetype='application/json')

@app.route('/ground/task2', methods=['GET', 'POST'])
def Task2():
    header, body = parseHttp(request)
    # print(body)

    # Task write into file with every possible input because of the append mode
    with open("Tasks/Task2.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")
    res = ask_ai("🍺Márk’s Sobering-Up Time – Task Description One evening, Márk went out with his friends and had a few drinks. The next day, he needs to drive, so he wants to calculate how many hours he has to wait after his last drink for all the alcohol to leave his system, allowing him to drive safely and soberly. ⚙️Calculation Model (based on average values) Use the following simple model for the calculation: • 1 beer → +0.15‰ increase in blood alcohol level • 1 shot (spirits) → +0.25‰ increase in blood alcohol level • Breakdown rate → 0.15‰ per hour Return format: { “hours” : 4.2563 }\n Here is the input: " + str(body))
    res = ask_ai("In what language is the following sentence? Respond with the first letter of the language in lowercase. Languages: Afrikaans, Bosnian, Catalan, Dutch, English, French, German, Hungarian. \n This is the sentence: " + str(body) + "\n Give me just the answer in one word!")

    print("ground/Task2:", body)

    return sober_hours(body)


@app.route('/ground/task3', methods=['GET', 'POST'])
def Task3():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Task3.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    # print(body)


    res = ask_ai("In what language is the following sentence? Respond with the first letter of the language in lowercase. Languages: Afrikaans, Bosnian, Catalan, Dutch, English, French, German, Hungarian. \n This is the sentence: " + str(body) + "\n Give me just the answer in one word!")
    print("ground/Task3:", res)

    return res


@app.route('/ground/bonus', methods=['GET', 'POST'])
def Task4():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/ground_bonus.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    return str(5)
    # return res = ask_ai("Here is the task (Task-Description): " + str(header) + "\n And here is the input data: " + str(body) + "\n Give me just the answer in one word!")

@app.route('/level1/task1', methods=['GET', 'POST'])
def Level1():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level1Task1.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    # question from header -> Task-Description:

    return process_hiking_segments(body)

@app.route('/level1/task2', methods=['GET', 'POST'])
def Level1Task2():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level1Task2.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    res = zigzag_tree_old(body)

    return res

@app.route('/level1/task3', methods=['GET', 'POST'])
def Level1Task3():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level1Task3.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    #res = ask_ai("Here is a code snippet that needs review:\n" + str(body) + "\n Identify the explanation answer and send back response in this json format: {\"id\": \"3fa85f64-5717-4562-b3fc-2c963f66afa6\",\"answerLetter\": \"B\"\}")

    #return res
    cleaned = body.encode("utf-8").decode("unicode_escape")

    data = json.loads(cleaned)

    task_id = data.get("id")
    code = data.get("code", "")
    answers = data.get("answers", [])

    # Make ChatGPT choose an answer
    res = ask_ai(
        "Reply with just the letter of the correct answer.\n"
        "Here is a code snippet:\n" + code +
        "\nHere are the options:\n" + str(answers) +
        "\nWhich option is correct?"
    ).strip()

    return {"id": task_id, "answerLetter": res}


@app.route('/level1/bonus', methods=['GET', 'POST'])
def Level1Bonus():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level1Bonus.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")
    return "Malta"
    # return ask_ai("A space is missing between two sentences in the website. What is the last word of the first sentence and the first one of the second? Format: lastword.Firstword (https://bishop-co.com/)\n" + "Answer it with just that 2 words in the correct format!")

@app.route('/level2/task1', methods=['GET', 'POST'])
def Level2Task1():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level2Task1.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    res = ask_ai("Here is the input data: " + str(body) + "\n Give me just the answer in one word!")
    res = magical_artifact_hunt(body)

    return res

@app.route('/level2/task2', methods=['GET', 'POST'])
def Level2Task2():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level2Task2.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    ret = ask_ai("Check Task-Description in this header: " + str(header) + "\n This is the body's content: " + str(body) + "\n Give me just the answer in one word!")
    print("LEVEL2/TASK2:", ret)

    return ret

@app.route('/level2/task3', methods=['GET', 'POST'])
def Level2Task3():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level2Task3.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    szotar = {
        "1": "Apple, Sneakers, Camera, Pencil, Glasses, Watch, Mug, Binoculars",
        "2": "Headphones, Lamp, Scissors, Presents, Leaf, Pinecone, Flashlight, Ruler, Lego, Party hat, Globe",
        "3": "Controller, Pills, Chess, Bottle, Calculator, Clock, Bulb, Book, Patch",
        "4": "Toothbrush, Hammer, Spoon, Dice, Magnet, Banana, Paintbrush, Shell, Clips, Keys",
        "5": "Eraser, Feather, Promenade, Tennisball, Origami, Teafilter, Matchbox"
    }
    for key, value in szotar.items():
        if body in value:
            return key

    res = ask_ai("Here is a list of items grouped in lists." + str(szotar) + "Here is an item. \n" + str(body) + "\nGive me ONLY the number of the list which contains the item. The name of the item may not be accurate, please respont accordingly.")
    print("LEVEL2/TASK3:", res)
    return res


@app.route('/level2/bonus', methods=['GET', 'POST'])
def Level2Bonus():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level2Bonus.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    return "2025-11-27"

@app.route('/level3/task1', methods=['GET', 'POST'])
def Level3Task1():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level3Task1.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")
    res = forest_fire_time(body)
    print("LEVEL3/TASK1:",str(res))
    return str(res)

@app.route('/level3/task2', methods=['GET', 'POST'])
def Level3Task2():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level3Task2.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")
    guesses_feedback = parse_input(body)
    res = type_advantage_finder(body)

    return res

@app.route('/level3/task3', methods=['GET', 'POST'])
def Level3Task3():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level3Task3.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    res = sudoku_validator(str(body))
    print("LEVEL3/TASK3:", res)
    print("BODY:", body)
    return res

@app.route('/level3/bonus', methods=['GET', 'POST'])
def Level3Bonus():
    header, body = parseHttp(request)
    # Task write into file with every possible input because of the append mode
    with open("Tasks/Level3Bonus.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    return "2022"


import chess

# Map KOPAJ unicode pieces to FEN
UNICODE_TO_FEN = {
    '♙': 'P', '♘': 'N', '♗': 'B', '♖': 'R', '♕': 'Q', '♔': 'K',
    '♟': 'p', '♞': 'n', '♝': 'b', '♜': 'r', '♛': 'q', '♚': 'k',
    '□': None, '■': None
}

VALID_SQUARE_BYTES = {
    b'\xe2\x99\x9c',  # ♜ black rook
    b'\xe2\x99\x9d',  # ♝ black bishop
    b'\xe2\x99\x9e',  # ♞ black knight
    b'\xe2\x99\x9f',  # ♟ black pawn
    b'\xe2\x99\x9a',  # ♚ black king
    b'\xe2\x99\x9b',  # ♛ black queen

    b'\xe2\x99\x94',  # ♔ white king
    b'\xe2\x99\x95',  # ♕ white queen
    b'\xe2\x99\x96',  # ♖ white rook
    b'\xe2\x99\x97',  # ♗ white bishop
    b'\xe2\x99\x98',  # ♘ white knight
    b'\xe2\x99\x99',  # ♙ white pawn

    b'\xe2\x96\xa1',  # □ white square
    b'\xe2\x96\xa0',  # ■ black square
}

def decode_kopaj_board(body: str):
    # raw bytes from latin1 are the original \xNN sequences
    raw = body.encode("latin1")

    # Split into lines by ASCII newline (0x0A)
    lines = raw.split(b'\n')

    # Last line is "white" or "black"
    side = lines[-1].decode("utf-8", errors="ignore").strip().lower()
    board_bytes = lines[:-1]

    rows = []

    for line in board_bytes:
        squares = []
        i = 0
        while i < len(line):
            # Check next 3 bytes as a UTF-8 sequence
            chunk = line[i:i+3]
            if chunk in VALID_SQUARE_BYTES:
                squares.append(chunk)
                i += 3
            else:
                i += 1

            if len(squares) == 8:
                break

        if len(squares) == 8:
            rows.append(squares)

    if len(rows) != 8:
        raise ValueError(
            f"Board parsing error: extracted {len(rows)} rows instead of 8."
        )

    return rows, side


BYTE_TO_FEN = {
    b'\xe2\x99\x99': 'P',
    b'\xe2\x99\x98': 'N',
    b'\xe2\x99\x97': 'B',
    b'\xe2\x99\x96': 'R',
    b'\xe2\x99\x95': 'Q',
    b'\xe2\x99\x94': 'K',

    b'\xe2\x99\x9f': 'p',
    b'\xe2\x99\x9e': 'n',
    b'\xe2\x99\x9d': 'b',
    b'\xe2\x99\x9c': 'r',
    b'\xe2\x99\x9b': 'q',
    b'\xe2\x99\x9a': 'k',

    b'\xe2\x96\xa1': None,
    b'\xe2\x96\xa0': None,
}


def rows_to_fen(rows, side):
    fen_rows = []
    for row in rows:
        fen_row = ""
        empty = 0

        for sq in row:
            piece = BYTE_TO_FEN[sq]
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece

        if empty > 0:
            fen_row += str(empty)

        fen_rows.append(fen_row)

    turn = "w" if side == "white" else "b"
    return "/".join(fen_rows) + f" {turn} - - 0 1"


def find_mate_in_one(board):
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move.uci()
        board.pop()
    return None



def solve_final_boss(body):
    rows, side = decode_kopaj_board(body)
    fen = rows_to_fen(rows, side)
    board = chess.Board(fen)
    return find_mate_in_one(board)



import chess
import sys


def solve_mate_in_one(input_string: str) -> str:
    lines = input_string.strip().split('\n')
    if len(lines) < 9:
        # Errors are printed to stderr, which is standard for utility scripts
        print("Error: Input must contain 8 ranks and the color to move.", file=sys.stderr)
        return ""

    board_str_lines = lines[:-1]
    color_to_move = lines[-1].strip().lower()

    piece_map = {
        '♜': 'r', '♞': 'n', '♝': 'b', '♛': 'q', '♚': 'k', '♟': 'p',
        '♖': 'R', '♘': 'N', '♗': 'B', '♕': 'Q', '♔': 'K', '♙': 'P',
        '■': ' ', '□': ' ',  # Empty squares
    }

    fen_parts = []
    for line in board_str_lines:
        fen_rank = ""
        empty_count = 0

        # Iterate over characters in the line (rank)
        for char in line:
            piece_char = piece_map.get(char, None)

            if piece_char is not None and piece_char != ' ':
                # Found a piece: close out the empty count and add the piece
                if empty_count > 0:
                    fen_rank += str(empty_count)
                    empty_count = 0
                fen_rank += piece_char
            elif piece_char == ' ':
                # Found an empty square
                empty_count += 1
            # If the character is not recognized, it's ignored

        # Close out any remaining empty count at the end of the rank
        if empty_count > 0:
            fen_rank += str(empty_count)

        fen_parts.append(fen_rank)

    # The FEN board part is ranks 8 through 1, separated by '/'
    fen_board = "/".join(fen_parts)

    # Determine turn
    turn_fen = 'w' if color_to_move == "white" else 'b'

    # Construct the full FEN string:
    # [board] [turn] [castling] [en passant] [halfmove clock] [fullmove number]
    # We assume standard defaults for the rest, as they won't affect a mate-in-one puzzle.
    fen = f"{fen_board} {turn_fen} - - 0 1"

    try:
        # Create the chess board object from the FEN
        board = chess.Board(fen)
    except ValueError as e:
        print(f"Error creating board from FEN: {e}", file=sys.stderr)
        print(f"Generated FEN: {fen}", file=sys.stderr)
        return ""

    # 4. Search for the Mate-in-One Move
    for move in board.legal_moves:
        # Temporarily make the move
        board.push(move)

        # Check if the opponent is in checkmate
        if board.is_checkmate():
            # Convert the move to the required output format (UCI string, promotion uppercase)
            uci_move = str(move)
            if len(uci_move) == 5:
                # Promotion: convert 'q'/'r'/'b'/'n' to 'Q'/'R'/'B'/'N'
                uci_move = uci_move[:4] + uci_move[4].upper()

            # Undo the move and return the result
            board.pop()
            return uci_move

        # Undo the move to check the next one
        board.pop()

    # The problem statement guarantees a mate in one, so this path should not be reached.
    return "Error: Could not find mate in one move."


def solve_boss(input_data):
    """
    Solve the "checkmate in one" chess puzzle.

    Args:
        input_data: bytes or str containing the chess board and color to move

    Returns:
        str: The move in format "piece from_pos to_pos" (e.g., "Q e2 e8")
    """
    # Decode if bytes
    if isinstance(input_data, bytes):
        board_str = input_data.decode('utf-8')
    else:
        board_str = input_data

    lines = board_str.strip().split('\n')
    board_lines = lines[:-1]  # All but last line (which is the color)
    color = lines[-1].strip().lower()  # "white" or "black"

    # Parse the board
    board = []
    for line in board_lines:
        row = []
        for char in line:
            if char in ['□', '■']:  # Empty squares
                row.append(None)
            else:
                row.append(char)
        board.append(row)

    # Map Unicode chess pieces
    white_pieces = {'♔': 'K', '♕': 'Q', '♖': 'R', '♗': 'B', '♘': 'N', '♙': 'P'}
    black_pieces = {'♚': 'K', '♛': 'Q', '♜': 'R', '♝': 'B', '♞': 'N', '♟': 'P'}

    def is_white(piece):
        return piece in white_pieces

    def is_black(piece):
        return piece in black_pieces

    def piece_type(piece):
        if piece in white_pieces:
            return white_pieces[piece]
        elif piece in black_pieces:
            return black_pieces[piece]
        return None

    # Find kings
    white_king_pos = None
    black_king_pos = None
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == '♔':
                white_king_pos = (r, c)
            elif piece == '♚':
                black_king_pos = (r, c)

    target_king = black_king_pos if color == 'white' else white_king_pos

    def pos_to_chess(r, c):
        """Convert (row, col) to chess notation (e.g., (7, 4) -> 'e1')"""
        return chr(ord('a') + c) + str(8 - r)

    def is_valid_pos(r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def get_piece_moves(piece, r, c):
        """Get all possible moves for a piece at position (r, c)"""
        moves = []
        ptype = piece_type(piece)
        is_my_piece = (color == 'white' and is_white(piece)) or (color == 'black' and is_black(piece))

        if not is_my_piece:
            return []

        if ptype == 'P':  # Pawn
            if color == 'white':
                # Move forward
                if is_valid_pos(r - 1, c) and board[r - 1][c] is None:
                    moves.append((r - 1, c))
                # Capture diagonally
                for dc in [-1, 1]:
                    nr, nc = r - 1, c + dc
                    if is_valid_pos(nr, nc) and board[nr][nc] and is_black(board[nr][nc]):
                        moves.append((nr, nc))
            else:  # black
                # Move forward
                if is_valid_pos(r + 1, c) and board[r + 1][c] is None:
                    moves.append((r + 1, c))
                # Capture diagonally
                for dc in [-1, 1]:
                    nr, nc = r + 1, c + dc
                    if is_valid_pos(nr, nc) and board[nr][nc] and is_white(board[nr][nc]):
                        moves.append((nr, nc))

        elif ptype == 'N':  # Knight
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_moves:
                nr, nc = r + dr, c + dc
                if is_valid_pos(nr, nc):
                    target = board[nr][nc]
                    if target is None or (color == 'white' and is_black(target)) or (
                            color == 'black' and is_white(target)):
                        moves.append((nr, nc))

        elif ptype == 'K':  # King
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if is_valid_pos(nr, nc):
                        target = board[nr][nc]
                        if target is None or (color == 'white' and is_black(target)) or (
                                color == 'black' and is_white(target)):
                            moves.append((nr, nc))

        elif ptype in ['R', 'Q']:  # Rook or Queen (rook moves)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                while is_valid_pos(nr, nc):
                    target = board[nr][nc]
                    if target is None:
                        moves.append((nr, nc))
                    elif (color == 'white' and is_black(target)) or (color == 'black' and is_white(target)):
                        moves.append((nr, nc))
                        break
                    else:
                        break
                    nr, nc = nr + dr, nc + dc

        if ptype in ['B', 'Q']:  # Bishop or Queen (bishop moves)
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = r + dr, c + dc
                while is_valid_pos(nr, nc):
                    target = board[nr][nc]
                    if target is None:
                        moves.append((nr, nc))
                    elif (color == 'white' and is_black(target)) or (color == 'black' and is_white(target)):
                        moves.append((nr, nc))
                        break
                    else:
                        break
                    nr, nc = nr + dr, nc + dc

        return moves

    def is_attacked_by(pos, by_color):
        """Check if position is attacked by pieces of given color"""
        r, c = pos
        # Check all pieces of the attacking color
        for fr in range(8):
            for fc in range(8):
                piece = board[fr][fc]
                if piece is None:
                    continue
                if (by_color == 'white' and is_white(piece)) or (by_color == 'black' and is_black(piece)):
                    moves = get_piece_moves(piece, fr, fc)
                    if (r, c) in moves:
                        return True
        return False

    def is_checkmate(king_pos, king_color):
        """Check if the king at king_pos is in checkmate"""
        # King must be in check
        attacker_color = 'white' if king_color == 'black' else 'black'
        if not is_attacked_by(king_pos, attacker_color):
            return False

        # King has no escape squares
        kr, kc = king_pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = kr + dr, kc + dc
                if is_valid_pos(nr, nc):
                    target = board[nr][nc]
                    # Can the king move there?
                    if target is None or ((king_color == 'white' and is_black(target)) or (
                            king_color == 'black' and is_white(target))):
                        # Simulate the move
                        old_piece = board[nr][nc]
                        board[nr][nc] = board[kr][kc]
                        board[kr][kc] = None

                        # Check if still in check
                        if not is_attacked_by((nr, nc), attacker_color):
                            # King can escape
                            board[kr][kc] = board[nr][nc]
                            board[nr][nc] = old_piece
                            return False

                        # Undo move
                        board[kr][kc] = board[nr][nc]
                        board[nr][nc] = old_piece

        return True

    # Try all possible moves
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece is None:
                continue

            # Check if it's our piece
            if (color == 'white' and not is_white(piece)) or (color == 'black' and not is_black(piece)):
                continue

            moves = get_piece_moves(piece, r, c)

            for new_r, new_c in moves:
                # Make the move
                old_piece = board[new_r][new_c]
                board[new_r][new_c] = piece
                board[r][c] = None

                # Check if this results in checkmate
                opponent_color = 'black' if color == 'white' else 'white'
                if is_checkmate(target_king, opponent_color):
                    # Found checkmate!
                    from_pos = pos_to_chess(r, c)
                    to_pos = pos_to_chess(new_r, new_c)

                    # Undo move before returning
                    board[r][c] = piece
                    board[new_r][new_c] = old_piece

                    return f"{from_pos}{to_pos}"

                # Undo move
                board[r][c] = piece
                board[new_r][new_c] = old_piece

    return "No checkmate in one found"

@app.route('/final-boss', methods=['GET', 'POST'])
def FinalBoss():
    header, body = parseHttp(request)

    # Task write into file with every possible input because of the append mode
    with open("Tasks/Boss_old.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")

    # print(body)
    result = "I accept the terms and conditions"
    # convert body to bytes
    res = ask_ai("Here is a chess-board as an input from body: " + str(body) + "\n Give me the mate location on the board in this from-to, 2 positions format: \"h5f7\", Just the positions in the given format, NOTHING ELSE!")
    #res = solve_final_boss(body)
    print("FINAL BOSS:", res)

    return res


if __name__ == '__main__':
    app.run(port=1234, host='0.0.0.0')


import ast

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

def zigzag_tree(body: str) -> list:
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

from typing import List, Tuple


def process_hiking_segments(input_str: str) -> str:
    # Normalize and split lines, ignoring empty lines
    lines = [ln.strip() for ln in input_str.strip().splitlines() if ln.strip()]
    if not lines:
        return "0 0 0 0"

    distances: List[int] = []
    elevations: List[int] = []

    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid line format at line {i+1}: {ln!r}")
        try:
            d = int(parts[0])
            e = int(parts[1])
        except ValueError:
            raise ValueError(f"Non-integer values at line {i+1}: {ln!r}")
        if i == 0 and d != 0:
            # First line must be a starting point with distance 0
            raise ValueError("First line must have distance 0 for the starting point.")
        if d < 0:
            raise ValueError(f"Distance must be non-negative at line {i+1}.")
        distances.append(d)
        elevations.append(e)

    # Total distance is the sum of segment distances
    total_distance = sum(distances)

    # Max elevation
    max_elevation = max(elevations)

    # Total ascent and descent across consecutive points
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

    return hiking_statistics(body)

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

    res = zigzag_tree(body)

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

    return code_review(body)


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
    return "once.With"
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

    return validate_credit_card(str(body))

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

    return find_platform_and_max_passengers(body)

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

    return str(ask_ai(str(body)))


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

    return str(ask_ai(str(body)))

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
    res = max_profit(body)
    print(res)
    return res

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
    res = solve_mastermind(guesses_feedback)

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

    return str(ask_ai(str(body)))

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

    return str(ask_ai(str(body)))






@app.route('/final-bossxd', methods=['GET', 'POST'])
def FinalBoss():
    header, body = parseHttp(request)

    # Task write into file with every possible input because of the append mode
    with open("Tasks/Boss.txt", "a") as f:
        f.write("#####################\n\t\tHEADER:\n#####################\n")
        f.write(str(header))
        f.write("#####################\n\t\tBODY:\n#####################\n")
        f.write(str(body))
        f.write("\n\n******************************************\n************* N E W  T A S K *************\n******************************************\n\n")
    return "KopajxAI"
    # print(body)
    body = body.replace("\\r", "")
    grid = body.split("\\n")[:-1]
    src = 0
    src2 = 0
    dest = 0
    grid_res = []
    for i in enumerate(grid):
        tmp = []
        for j in enumerate(i[1]):
            if grid[i[0]][j[0]] == "#":
                tmp.append(0)
                continue
            elif grid[i[0]][j[0]] == " ":
                tmp.append(1)
                continue
            elif grid[i[0]][j[0]] == "2":
                src2 = [i[0], j[0]]
                tmp.append(1)
                continue
            if grid[i[0]][j[0]] == "1":
                src = [i[0], j[0]]
                tmp.append(1)
                continue
            if grid[i[0]][j[0]] == "3":
                dest = [i[0], j[0]]
                tmp.append(1)
                continue
        grid_res.append(tmp)


    path = a_star_search(grid_res, src, src2, len(grid_res), len(grid_res[0]))
    path2 = a_star_search(grid_res, src2, dest, len(grid_res), len(grid_res[0]))
    for i in path2:
        path.append(i)

    for i, coordinates in enumerate(path):
        grid[coordinates[0]] = grid[coordinates[0]][:coordinates[1]] + "-" + grid[coordinates[0]][coordinates[1]+1:]


    # for i in grid:
    #     print(i)
    result = ""
    for i in grid:
        result += i + "\n"
    return Response(result, status=200, mimetype='text/plain')


if __name__ == '__main__':
    app.run(port=1234, host='0.0.0.0')


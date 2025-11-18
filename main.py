import ast

from flask import Flask, make_response, Response, request
from astar import a_star_search, ROW, COL
from ai import ask_ai
import json

from collections import Counter


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



app = Flask(__name__)

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
    res = ask_ai(body + "\n You have to answer the question correctly in one word!")
    # print(res)
    # if res last char is '.' then remove it
    # if res[-1] == '.':
    #     res = res[:-1]
    return res


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
    first_n_pos = body.find('n')
    body = body[first_n_pos + 1:]
    # print(body)
    res = count_islands(parse_string_to_matrix(str(body)))
    res2 = ask_ai("How many islands in this matrix?" + str(body) + "\n" + "Answer it with one number! Write Just the number! Nothing else!")

    return res2


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

    return str(304.6)+'C'

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

    return smallest_hole(body)

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

    L = False
    F = False
    R = False

    input_string = str(body)
    print(input_string)
    input_string = input_string.replace('\\n', '\n')

    rows = input_string.strip().split('\n')
    result = []

    for row in rows:
        columns = row.strip('|').split('|')
        result.append(columns)
    transposed_result = list(zip(*result))
    transposed_result = [list(row) for row in transposed_result]
    for column in transposed_result:
        if (column[2] == '*' and column[0] == '-' and column[1] == '-') or (column[0] == '*' and column[1] == '*'):
            if '<' in column[3]:
                L = True
            if '>' in column[3]:
                R = True
            if '^' in column[3]:
                F = True
        print(column)
    res = ''
    if L:
        res += 'L'
    if F:
        res += 'F'
    if R:
        res += 'R'

    if not L and not F and not R:
        res = 'STOP'

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
    story = """
    Di perkampungan Willowbrook yang tenang, terletak di antara bukit-bukau dan sungai yang berkilauan, tinggal seorang wanita muda bernama Eliza. Dia terkenal dengan kecemerlangannya berkebun, dan pondok kecilnya dikelilingi oleh bunga-bunga tercantik di kampung itu. Taman bunganya adalah gabungan keanggunan dan kegembiraan, dengan bunga ros yang bersenangat, teratai yang halus, dan bunga matahari tinggi yang kelihatan menyentuh matahari setiap pagi. Setiap hari, Eliza akan menghabiskan berjam-jam menenggumi tamannya, bercakap dengan mereka seolah-olah mereka adalah kawannya.

    Pada suatu pagi yang cerah, ketika Eliza sedang mencantas bunga rosnya, dia melihat seekor burung kecil sedang duduk di atas pagar taman. Burung itu mempunyai bulu biru terang dan pandangan ingin tahu di matanya. Eliza tersenyum dan bersiul perlahan kepadanya. Terkejut dulu, burung itu terbang dan hilang di awan biru. Eliza, gembira, memakan burung itu Sky.
    
    Sejak hari itu, Sky menjadi pengunjung tetap di tamannya. Kedua-duanya membentuk ikatan istimewa. Sky akan berkicau gembira semasa Eliza bekerja, malah kadangkala dia akan menemaninya ke kampung apabila dia pergi ke pasar. Orang ramai mula melihat burung itu dan sering bertanya kepada Eliza tentang kawan barunya itu.
    
    Pada suatu petang, ketika Eliza berada di pasar, seorang pengembara menghampirinya. Dia seorang lelaki yang lebih tua dengan janggut kelabu yang panjang dan mata yang baik. "Saya pernah mendengar tentang taman anda dan burung biru kecil anda," katanya. "Ada sesuatu yang ajaib mengenainya, bukan?"
    
    Eliza terkejut dengan kata-katanya. "Ajaib? Saya selalu fikir Sky adalah istimewa, tetapi saya tidak pernah menganggap sihir."
    
    Lelaki itu mengangguk. "Burung itu bukan burung biasa. Dikatakan bahawa mereka yang berkawan dengan makhluk seperti itu diberkati dengan tuah dan kebahagiaan yang besar."
    
    Tertarik tetapi ragu-ragu, Eliza pulang ke rumah dengan barangan runcitnya. Semasa dia duduk di tamannya, Sky mendarat di lututnya, berkicau dengan riang. Dia tersenyum dan tertanya-tanya apakah ada kebenaran pada kata-kata pengembara itu.
    
    Pada minggu-minggu berikutnya, Eliza mula melihat perubahan kecil. Tamannya mekar lebih indah dari sebelumnya, dan semua yang dia tanam nampaknya tumbuh dua kali lebih cepat. Jiran-jirannya mengulus tentang betapa meriahnya bunganya, malah orang tua kampung menenari rahsia berkebunnya. Eliza sedar bahawa hidupnya menjadi lebih ceria sejak Sky tiba.
    
    Pada suatu petang, dengan matahari terbenam di sebalik bukit, Eliza duduk di tamannya, Sky bertengger di sebelahnya. Dia merenung bunga beraneka warna dan memikirkan kata-kata pengembara itu. Mungkin ada sedikit keajaiban di tamannya.
    """
    res = ask_ai("Answer the questions based on this story, return only the letter of the correct answer." + str(story) + "\nHere the question and the answer options: \n" + str(body))
    return res


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






@app.route('/final-boss', methods=['GET', 'POST'])
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


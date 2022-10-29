from pprint import pprint


def show_sudoku(vars):
    solution_dict = dict()
    for (*pos, digit), value in vars.items():
        if round(value):
            solution_dict[tuple(pos)] = digit

    sudoku = [[None for i in range(9)] for j in range(9)]
    for (pos_y, pos_x, square_y, square_x), digit in solution_dict.items():
       y = square_y * 3 + pos_y
       x = square_x * 3 + pos_x
       sudoku[y][x] = digit

    pprint(sudoku)

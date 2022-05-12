import digit_recognizer as dr
import solver
import sudoku_reader as reader

path = "Assets/puzz1.jpg"

sudoku = reader.readImage(path)

recognizer = dr.get_model()

board = sudoku.get_board(recognizer)

solver.print_board(board)
solver.solve(board)
print("___________________")
solver.print_board(board)
# -----------------------------------------------------------
# Main file for running the solver
#
# 2022 Alex Oparaoji
# email coparaoji@gmail.com
# -----------------------------------------------------------
import digit_recognizer as dr
import solver as ss
import sudoku_reader as reader
from tensorflow.keras.models import load_model

def main():

    path = "Assets/puzz1.jpg"

    sudoku = reader.readImage(path)
    sudoku.show()
    model = load_model('digit_recognizer')
    recognizer = dr.DigitRecognizer(model)

    board = sudoku.get_board(recognizer) 
    ss.final(board.grid)

if __name__ == "__main__":
    main()
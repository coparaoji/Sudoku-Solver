# -----------------------------------------------------------
# Main file for running the solver
#
# 2022 Alex Oparaoji
# email coparaoji@gmail.com
# -----------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import digit_recognizer as dr
import solver as ss
import sudoku_reader as reader
from tensorflow.keras.models import load_model

def main():

    path = "Assets/puzz1.jpg"

    # Read the image.
    sudoku = reader.readImage(path) #loads the image, assigns a name, and Image.show,InputImage.getBaord functions
    sudoku.show()

    # Get the model.
    model = None #broken need to retrain model and output newer model format - load_model('digit_recognizer')

    recognizer = dr.DigitRecognizer(model)

    # Attempting to recognize digits.
    board = sudoku.get_board(recognizer)

    # Solving the sudoku puzzle from gotten form the image..
    ss.final(board.grid)

if __name__ == "__main__":
    main()
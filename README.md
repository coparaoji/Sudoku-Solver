# Sudoku-Solver
A project to solve sudoku puzzles and practice SOLID design principles.

#### Recognizing the Board
Using this bit from the OpenCV docs,  
*"Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition."*  
we can assume the board is the largest contour in the image and find the corners to extract the grid.  

Before we find the sontours some image preprocceing sters need to be done to ensure accuracy; greysclaing, blurring, thresholding, and dilation.  

Once the vertices of the contour have been retereved then we will do a K-means cluster for 4 clusters and use their centers as the 4 corners of the graph.  

#### Recognizing the numbers
Once the grid has been located we can roughly get the grid-boxes by evenly splitting the grid into 81 9x9 boxes. A convolutional neural network will be used to recognize the numbers. It has 4 convolutional layers, 2 pooling layers, and 2 fully-connected layers. The full architecture and training process can be found in [here]. 
#### Solving Sudoku

#### Results

#### Final words
1. It would have been more reliable to use a ConvNN to locate the board. That would however require a labeled dataset of sudoku images which I presently don't have.
2. Back tracking works to solve sudoku all the time but a more efficient method exist; [*insert method here*]. It doesn't work all the time but it could be used first then backtracking sould be a fallback option.

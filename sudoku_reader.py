# -----------------------------------------------------------
# Contains the classes for images, finding the board, and finding the gridboxes
#
# 2022 Alex Oparaoji
# email coparaoji@gmail.com
# -----------------------------------------------------------
from mimetypes import init
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


class Image:
    """A superclass to add the ability to show images on screen"""
    title = ""
    cv_image = None

    def __init__(self, title) -> None:
        self.title = title

    def show(self):
        # This method shows the image on screen
        if(self.cv_image is not None):
            title = self.title
            screen_res = 1280, 720  #define the screen resalution
            scale_width = screen_res[0] / self.cv_image.shape[1]
            scale_height = screen_res[1] / self.cv_image.shape[0]
            scale = min(scale_width, scale_height)

            #resize window width and height
            window_width = int(self.cv_image.shape[1] * scale)
            window_height = int(self.cv_image.shape[0] * scale)

            
            #cv.WINDOW_NORMAL makes the output window resizealbe
            cv.namedWindow(title, cv.WINDOW_NORMAL)

            #resize the window according to the screen resolution
            cv.resizeWindow(title, window_width, window_height)

            #Wait for a key to be pressed before closing the window
            cv.imshow(title, self.cv_image)
            cv.waitKey(0) 
            cv.destroyAllWindows()

class InputImage(Image):
    # This is for handling the source image

    board = None
    path = None

    def __init__(self, title, path) -> None:
        super().__init__(title)
        self.path = path
        self.cv_image = cv.imread(path,0)

    
    def get_board(self, model):
        '''After making a copy, this method uses that copy to apply
        image preprocessing techniques before using opncv's findContours()
        to find the largest box on the image. This box is then treated
        as the grid and after clustering its vertexes into 4 corners, it
        is used to create the Board object and get 81 (grid)boxes'''
        image = self.cv_image.copy()
        
        image = cv.GaussianBlur(image,(11,11),0)

        outerBox = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
        outerBox = cv.bitwise_not(outerBox)

        #now to make the lines thicker
        outerBox = cv.dilate(outerBox, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)),iterations=4)
        
        #Here are those contours
        contours, heirarchy = cv.findContours(outerBox, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        kmeans = KMeans(n_clusters=4,random_state=0).fit(contours[0][:,0,:])

        grid_corners = kmeans.cluster_centers_

        #Here I find and label the corners of the puzzle.
        image_corners_dict = {'topLeft':(0,0), 'topRight':(image.shape[1],0), 'bottomLeft':(0,image.shape[0]), 'bottomRight':(image.shape[1],image.shape[0])}
        grid_corners_dict = {r: grid_corners[np.argmin([np.linalg.norm(i-np.array(k)) for i in grid_corners])] for (r,k) in image_corners_dict.items()}
        
        #This block of code makes the puzzle the main focus of the image using cv.warp.
        x = int(min(image.shape[0],image.shape[1]) * 0.90)
        while not(x%9==0):
            x-=1
        pts1 = np.float32(list(grid_corners_dict.values()))
        pts2 = np.float32([(0,0),(x,0),(0,x),(x,x)])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        newImage = cv.warpPerspective(outerBox, matrix, image.shape)
        newImage = newImage[0:x,0:x]
        
        return Board('board',newImage,grid_corners, model)

class Board(Image):
    #This holds, the board corners, list of boxes objects, and 2d list of sudoku board box numbers.

    corners = None
    boxes = None
    solved = False
    grid = []

    def __init__(self, title, cv_image, corners, model) -> None:
        #This is an initializer with a model so that digit predictions can be made.

        super().__init__(title)
        self.cv_image = cv_image
        self.corners = corners
        self.boxes = []

        #Here I make the gidbox objects and store them.
        rows = np.vsplit(cv_image,9)
        for r,r_image in enumerate(rows):
            cols= np.hsplit(r_image,9)
            for c, box in enumerate(cols):
                self.boxes.append(Box(f'row {r}, col {c}', box))
        
        #Here I make the grid and use the model to make predictions.
        self.grid = []
        m  = []
        for ind, i in enumerate(self.boxes):
            m.append(model.predict(i))
            if ((ind+1)%9==0):
                self.grid.append(m)
                m = []

class Box(Image):
    #This class is for storing grid-box images.
    value = -1
    
    def __init__(self, title, cv_image) -> None:
        super().__init__(title)
        self.cv_image = cv_image
    
def readImage(path):
    #Return an image object
    return InputImage(path,path)
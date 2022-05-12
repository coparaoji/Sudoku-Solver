from mimetypes import init
import cv2 as cv
import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans


class Image:
    title = ""
    cv_image = None

    def __init__(self, title) -> None:
        self.title = title

    def show(self):
        if(self.cv_image is not None):
            title = self.title
            #define the screen resulation
            screen_res = 1280, 720
            scale_width = screen_res[0] / self.cv_image.shape[1]
            scale_height = screen_res[1] / self.cv_image.shape[0]
            scale = min(scale_width, scale_height)

            #resized window width and height
            window_width = int(self.cv_image.shape[1] * scale)
            window_height = int(self.cv_image.shape[0] * scale)

            
            #cv.WINDOW_NORMAL makes the output window resizealbe
            cv.namedWindow(title, cv.WINDOW_NORMAL)

            #resize the window according to the screen resolution
            cv.resizeWindow(title, window_width, window_height)

            cv.imshow(title, self.cv_image)
            cv.waitKey(0)
            cv.destroyAllWindows()

class InputImage(Image):
    board = None
    path = None

    def __init__(self, title, path) -> None:
        super().__init__(title)
        self.path = path
        self.cv_image = cv.imread(path,0)

    '''After making a copy, this method uses that copy to apply
    image preprocessing techniques before using opncv's findContours()
    to find the lardgest bax on the image. This box is then treated
    as the grid and after clustering its vertexes into 4 corners, it
    is used to create the Board object'''
    def get_board(self, model):
        image = self.cv_image.copy()
        
        image = cv.GaussianBlur(image,(11,11),0)

        outerBox = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
        outerBox = cv.bitwise_not(outerBox)

        outerBox = cv.dilate(outerBox, cv.getStructuringElement(cv.MORPH_CROSS,(3,3)),iterations=4)
        
        contours, heirarchy = cv.findContours(outerBox, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        kmeans = KMeans(n_clusters=4,random_state=0).fit(contours[0][:,0,:])

        grid_corners = kmeans.cluster_centers_
        image_corners_dict = {'topLeft':(0,0), 'topRight':(image.shape[1],0), 'bottomLeft':(0,image.shape[0]), 'bottomRight':(image.shape[1],image.shape[0])}
        grid_corners_dict = {r: grid_corners[np.argmin([np.linalg.norm(i-np.array(k)) for i in grid_corners])] for (r,k) in image_corners_dict.items()}
        
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
    corners = None
    boxes = None
    solved = False
    grid = []

    def __init__(self, title, cv_image, corners, model) -> None:
        super().__init__(title)
        self.cv_image = cv_image
        self.corners = corners
        self.boxes = []
        rows = np.vsplit(cv_image,9)
        for r,r_image in enumerate(rows):
            cols= np.hsplit(r_image,9)
            for c, box in enumerate(cols):
                self.boxes.append(Box(f'row {r}, col {c}', box))
        self.grid = []
        m  = []
        for ind, i in enumerate(self.boxes):
            m.append(model.predict(i))
            if ((ind+1)%9==0):
                self.grid.append(m)
                m = []
        
        #todo get boxes from corners
    
class Box(Image):
    value = -1
    
    def __init__(self, title, cv_image) -> None:
        super().__init__(title)
        self.cv_image = cv_image
    
def readImage(path):
    return InputImage(path,path)
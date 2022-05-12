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
    def getBoard(self):
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
        
        return Board('board',newImage,grid_corners)




class Board(Image):
    corners = None
    boxes = None
    solved = False

    def __init__(self, title, cv_image, corners) -> None:
        super().__init__(title)
        self.cv_image = cv_image
        self.corners = corners
        self.boxes = []
        rows = np.vsplit(cv_image,9)
        for r,r_image in enumerate(rows):
            cols= np.hsplit(r_image,9)
            for c, box in enumerate(cols):
                self.boxes.append(Box(f'row {r}, col {c}', box))
        #todo get boxes from corners
    
class Box(Image):
    value = -1
    
    def __init__(self, title, cv_image) -> None:
        super().__init__(title)
        self.cv_image = cv_image
    
class DigitRecognizer:
    model = None


    def __init__(self, model) -> None:
        self.model = model

    #private method for getting the mask
    def getMask(self, x:np.ndarray):
        '''create a mask that iteratively makes sure that the borders
        are all gonna be removed
        '''
        test = x.copy()

        ymid = int(x.shape[0]/2) #.shape returns in format (height,width)
        xmid = int(x.shape[1]/2)

        top = 0
        left= 0 #for point (left,top)
        right= x.shape[0]-1
        bottom = x.shape[1]-1 #for point (right,bottom)

        '''I would like the go down and imaginary 5-pixe tick line 
        from the horizontal midpoint of the image, checking row by 
        row of the vertical line. Once every pixel in that
        row matches the supposed background of 0, then that
        would be where the topVariable should be. Doing the same
        for going up that line should yield bottom. Doing the same with
        a horizontal line should both the right and left variables'''
        #showImage(test[:,xmid-4:xmid+4])
        for index,i in enumerate(test[:,xmid-3:xmid+3]):
            all_zero = True
            for j in i:
                if not(j==0):
                    all_zero = False
            if(all_zero):
                top += index
                break

        for index,i in enumerate(np.transpose(test[xmid-3:xmid+3,:])):
            all_zero = True
            for j in i:
                if not(j==0):
                    all_zero = False
            if(all_zero):
                left += index
                break

        for index,i in enumerate(reversed(test[:,xmid-3:xmid+3])):
            all_zero = True
            for j in i:
                if not(j==0):
                    all_zero = False
            if(all_zero):
                bottom -= index
                break

        for index,i in enumerate(reversed(np.transpose(test[xmid-3:xmid+3,:]))):
            all_zero = True
            for j in i:
                if not(j==0):
                    all_zero = False
            if(all_zero):
                right -= index
                break



        mask = np.zeros(x.shape, dtype="uint8")
        cv.rectangle(mask, (left, top), (right, bottom), 255, -1)
        return mask


    #private preprocess for prediction
    def preprocess(self, test):
        mask = self.getMask(test)
        test = cv.bitwise_and(test, test, mask=mask)
        test = cv.bitwise_not(test)
        test = cv.resize(test,(32,32),interpolation=cv.INTER_AREA)
        test = cv.adaptiveThreshold(test,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
        test = cv.bitwise_not(test)
        contours, heirarchy = cv.findContours(test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros(test.shape,dtype=np.uint8)
        canvas2 = np.zeros(test.shape,dtype=np.uint8)
        cv.drawContours(canvas,contours,len(contours) - 1, (1), thickness = cv.FILLED)
        canvas2 = cv.bitwise_and(test,test,mask=canvas)
        test = canvas2
        test = cv.bitwise_not(test)
        
        #test = test.reshape((-1,32,32,1))
        return test
    #todo make image transformations and return a digit
    def predict(self, box: Box) -> int:
        
        img = box.cv_image
        x = self.preprocess(img)
        if(np.min(x[12:20,8:24]) == 255):
            return
        results = self.model.predict(x.reshape((-1,32,32,1)))
        results = np.argmax(results,axis = 1) + 1
        box.value = results
        return results

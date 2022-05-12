import cv2 as cv
import os
import numpy as np 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


def showImage(img,title='Resized Window'):

    #define the screen resulation
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    #resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    #cv.WINDOW_NORMAL makes the output window resizealbe
    cv.namedWindow(title, cv.WINDOW_NORMAL)

    #resize the window according to the screen resolution
    cv.resizeWindow(title, window_width, window_height)

    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

import random
#function to get a random file
def random_file(path):
    if(os.path.isfile(path)):
        return path
    else:
        x = os.listdir(path)
        return random_file(os.path.join(path,x[random.randint(0, len(x)-1)]))

#function to get random picture of specific number
def get_rand_pic_of(number):
    x = os.listdir("Dataset")
    y = x[random.randint(0, len(x)-1)]
    x = os.path.join("Dataset",y)
    x = os.path.join(x,str(number))
    return random_file(x)

#methods
#open image in opencv//done
#apply thresholding to image and resize//done
# and save the image in a new folder
#return an array of all the images for a given number and purpose (test or train)
import cv2 as cv
import os
import numpy as np

def open_image(path):
    x = cv.imread(path,0)
    if (x is None):
        raise Exception("File not found")
    return x

def preprocess(image):
    if(np.max(image)<100):
        image = image*255
    image =  cv.resize(image, (28,28), interpolation=cv.INTER_AREA)
    image =  cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15,10)
    image = cv.bitwise_not(image)
    image = image/255

    return image

def one_hot_encoder(num, num_classes = 9):
    x = np.zeros(num_classes)
    x[int(num)] = 1.0
    return x


class ImageRetriever:
    id = "" #file name
    label = "" #0-9
    purpose = "" #training_data or testing_data
    raw_image = None

    def __init__(self, raw_image = None, path = "", purpose = "", id = "")-> None:
        self.raw_image = raw_image
        self.path = path
        self.purpose = purpose
        self.id = id

    

    def get(self, path):
        self.raw_image = preprocess(open_image(path))
        self.id = path.split('/')[-1].split('.')[0]
        self.label = path.split('/')[-2]
        self.label = to_categorical(int(self.label)-1, num_classes=9)
        self.purpose = path.split('/')[1]

def get_data_split(file="training_data", image_retriever = ImageRetriever()):
    path = os.path.join("Dataset",file)
    train = np.ndarray((0,28,28,1))
    labels = np.ndarray((0,9))
    for number_folder in os.listdir(path):
        for number_image in os.listdir(os.path.join(path,number_folder)):
            image_path = os.path.join(path,number_folder,number_image)
            image_retriever.get(image_path)
            train = np.vstack([train,image_retriever.raw_image.reshape((1,28,28,1))])
            labels = np.vstack([labels, image_retriever.label.reshape((1,9))])
    return train, labels

def build_model():
    num_classes = 9

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3),
                    activation='relu',
                    input_shape=(28,28,1), padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="Adam",
              metrics=['accuracy'])

    return model

def build_and_train_model():
    model = build_model()
    train,labels = get_data_split()
    model.fit(train,labels, epochs = 5)

    test, test_labels = get_data_split(file = "testing_data")

    print(model.evaluate(test,test_labels))
    return model

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
        test = cv.resize(test,(28,28),interpolation=cv.INTER_AREA)
        test = cv.adaptiveThreshold(test,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,3,5)
        test = cv.bitwise_not(test)
        contours, heirarchy = cv.findContours(test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        canvas = np.zeros(test.shape,dtype=np.uint8)
        canvas2 = np.zeros(test.shape,dtype=np.uint8)
        cv.drawContours(canvas,contours,len(contours) - 1, (1), thickness = cv.FILLED)
        canvas2 = cv.bitwise_and(test,test,mask=canvas)
        test = canvas2
        #test = cv.bitwise_not(test)
        
        #test = test.reshape((-1,32,32,1))
        return test
    #todo make image transformations and return a digit
    def predict(self, box) -> int:
        
        img = box.cv_image
        x = self.preprocess(img)
        if(np.max(x[12:20,8:24]) == 0):
            return 0
        results = self.model.predict(x.reshape((-1,28,28,1)))
        results = np.argmax(results,axis = 1) + 1
        results = results[0]
        box.value = results
        return results

def get_model():
    model = load_model('saved_model/my_model')
    model = DigitRecognizer(model)
    return model

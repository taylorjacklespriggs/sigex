import cv2
import numpy as np

class VectorViewer:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.display = np.zeros((height, width), dtype=np.float32)
        self.idx = 0
    def show(self, vector):
        self.display[self.idx] = cv2.resize(np.mat(vector), (self.width,1), interpolation=cv2.INTER_NEAREST)
        self.idx = (self.idx+1)%self.height
        self.display[self.idx] = 1
        cv2.imshow(self.name, self.display)

def view_all(pause=30):
    cv2.waitKey(pause)


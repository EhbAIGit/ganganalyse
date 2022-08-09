import cv2

class Window:
    def __init__(self, name):
        self.name = name
        self.view = self.create_display()
    
    def create_display(self):
        return cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    
    def resize_display(self, size):
        cv2.resizeWindow(self.name, size)

    def display(self, image):
        cv2.imshow(self.name, image)
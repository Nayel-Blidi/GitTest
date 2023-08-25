import numpy as np
import os
from PIL import Image

class imageHandler:
    def __init__(self, 
                inputPath=None,    imagesNames=None, 
                imagesHeight=64,    imagesWidth=64
                ):
        
        self.inputPath =        inputPath
        self.imagesNames =      imagesNames
        
        self.imagesHeight =     imagesHeight
        self.imagesWidth =      imagesWidth
    
    def _getPath(self):
        filePath = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
        return filePath
    
    def getImagesNames(self):
        
        path = f"{imagesHandler._getPath(self)}\Images"
        all_files = os.listdir(path)
        
        if __name__ == "__main__":
            print(path)
            print(all_files)
        
        imagesNames = [file for file in all_files if file.lower().endswith(".jpg")]
        imagesNames = [file for file in all_files if file.lower().endswith(".png")]

        self.imagesNames = imagesNames
        return imagesNames
             
    def resizeImages(self):

        try:
            for image in self.imagesNames:
                
        except:
            print("No image found")
            
    
    def convolutionalSave(self, images):
        None
        
if __name__ == "__main__":
    class_test = imageHandler()
    class_test.getImagesNames()
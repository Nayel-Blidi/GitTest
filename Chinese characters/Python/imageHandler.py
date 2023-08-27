import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d
import sys
from tqdm import tqdm

class imageHandler:
    def __init__(self, 
                inputPath=False, outputPath=False,  
                imagesNames=[], 
                imagesHeight=64,    imagesWidth=64
                ):
        
        self.inputPath =        inputPath
        self.outputPath =       outputPath
        self.imagesNames =      imagesNames
        
        self.imagesHeight =     imagesHeight
        self.imagesWidth =      imagesWidth
    
    def _getPath(self):
        path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
        return path
    
    def getImagesNames(self):
        
        if self.inputPath == False:
            input_path = f"{imageHandler._getPath(self)}\Images"
        else:
            input_path = self.inputPath
        
        all_files = os.listdir(input_path)
        
        print("input path found :", input_path)
        if __name__ == "__main__" and (len(sys.argv) == 1):
            print("input files list generated (0:10) :", all_files[0:10])
            print(all_files)

        if [file for file in all_files if file.lower().endswith(".jpg")]:
            imagesNames = [file for file in all_files if file.lower().endswith(".jpg")]
            self.imagesNames = imagesNames
            
        if ([file for file in all_files if file.lower().endswith(".png")]):
            imagesNames = imagesNames.append([file for file in all_files if file.lower().endswith(".png")])
            self.imagesNames = imagesNames    

        if __name__ == "__main__":
            print("getImagesNames ended successfully")

        return imagesNames
             
    def resizeImages(self):

        if self.inputPath == False:
            input_path = f"{imageHandler._getPath(self)}\Images"
        else:
            input_path = self.inputPath

        if self.outputPath == False:
            output_path = f"{imageHandler._getPath(self)}\Images"
        else:
            output_path = self.outputPath
            
        if __name__ == "__main__" and (len(sys.argv) == 1):
            print(input_path)
            
        try:
            for image in tqdm(self.imagesNames):
                array = Image.open(f"{input_path}\{image}")
                array = array.resize( (self.imagesHeight, self.imagesWidth) )
                array = array.convert('L')
                array.save(f"{output_path}\{image}")  
        except:
            raise ValueError("No image found in resizeImages")
        
        if __name__ == "__main__":
            print("resizeImages ended successfully")

    def contrastImages(self):
        
        if self.inputPath == False:
            input_path = f"{imageHandler._getPath(self)}\Images"
        else:
            input_path = self.inputPath

        if self.outputPath == False:
            output_path = f"{imageHandler._getPath(self)}\Images"
        else:
            output_path = self.outputPath
            
        if __name__ == "__main__" and (len(sys.argv) == 1):
            print(input_path)

        try:
            for image in tqdm(self.imagesNames):
                array = np.array(Image.open(f"{input_path}\{image}"))
                array[array < 128] = 0
                array[array >= 128] == 255
                array = Image.fromarray(array).convert('L')
                array.save(f"{output_path}\{image}")  
        except:
            raise ValueError("No image found in contrastImages")
                
        if __name__ == "__main__":
            print("contrastImages ended successfully")        
    
    def convoluteImages(self):
        
        if self.inputPath == False:
            input_path = f"{imageHandler._getPath(self)}\Images"
        else:
            input_path = self.inputPath
            
        if self.outputPath == False:
            output_path = f"{imageHandler._getPath(self)}\Images"
        else:
            output_path = self.outputPath
        
        sobel_kernel_x = np.array( [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]] )
        sobel_kernel_y = sobel_kernel_x.T

        try:
            for image in tqdm(self.imagesNames):
                                
                array = Image.open(f"{input_path}\{image}")
                
                # Convolution with Sobel kernels for edge detection
                gradient_x = convolve2d(array, sobel_kernel_x, mode="same", boundary='fill', fillvalue=0)
                gradient_y = convolve2d(array, sobel_kernel_y, mode="same", boundary='fill', fillvalue=0)

                # Gradient magnitude
                gradient_magnitude = np.clip(np.sqrt(gradient_x**2 + gradient_y**2), 0,255).astype(np.uint8)
                gradient_magnitude = Image.fromarray(gradient_magnitude)
                gradient_magnitude.save(f"{output_path}\{image}")  
        except:
            raise ValueError("No image found in convoluteImages")
        
        if __name__ == "__main__":
            print("convoluteImages ended successfully")
            
    def rotateImages(self):

        if self.inputPath == False:
            input_path = f"{imageHandler._getPath(self)}\Images"
        else:
            input_path = self.inputPath

        if self.outputPath == False:
            output_path = f"{imageHandler._getPath(self)}\Images"
        else:
            output_path = self.outputPath
            
        if __name__ == "__main__" and (len(sys.argv) == 1):
            print(input_path)
            
        try:
            for image in tqdm(self.imagesNames):
                array = np.array(Image.open(f"{input_path}\{image}"))
                array = np.rot90(array, k=np.random.randint(-1, 2))
                array = Image.fromarray(array).convert('L')
                array.save(f"{output_path}\{image}")  
        except:
            raise ValueError("No image found in rotateImages")
        
        if __name__ == "__main__":
            print("rotateImages ended successfully")        
            

if __name__ == "__main__" and (len(sys.argv) == 1):
    class_test = imageHandler()
    class_test.getImagesNames()
    class_test.resizeImages()
    class_test.convoluteImages()
    

if __name__ == "__main__" and ( "conv_images" in sys.argv ):
    inputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\data"
    outputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\convoluted data"
    
    conv_image_class = imageHandler(inputPath=inputPath, outputPath=outputPath)
    conv_image_class.getImagesNames()
    conv_image_class.resizeImages()
    conv_image_class.convoluteImages()
    conv_image_class.contrastImages()
    
if __name__ == "__main__" and ( "contrast_images" in sys.argv ):
    inputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\data"
    outputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\contrasted data"
    
    contrast_image_class = imageHandler(inputPath=inputPath, outputPath=outputPath)
    contrast_image_class.getImagesNames()
    contrast_image_class.resizeImages()
    contrast_image_class.contrastImages()
    
if __name__ == "__main__" and ( "rotate_images" in sys.argv ):
    inputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\data"
    outputPath = "D:\Machine Learning\Chinese project\handwritten chinese numbers\\rotated data"
    
    rotate_image_class = imageHandler(inputPath=inputPath, outputPath=outputPath)
    rotate_image_class.getImagesNames()
    rotate_image_class.resizeImages()
    rotate_image_class.contrastImages()
    rotate_image_class.rotateImages()
    

import numpy as np
import cv2
from PIL import Image
import logging


#***********************************************************************
# Constants
#**********************************************************************
droste_scale = 256
TWO_PI_i  = 2j * np.pi
alpha     = (TWO_PI_i + np.log(droste_scale)) / TWO_PI_i
alpha_inv = TWO_PI_i / (TWO_PI_i + np.log(droste_scale))


#***********************************************************************
# Mapping functions
#**********************************************************************

def cartesian_to_complex(x,y):
    '''
    :param x: x-coordinates in Cartesian plane
    :param y: y-coordinates in Cartesian plane
    :return: one grid representing the complex plane
    '''
    Z = x + y * 1j
    return Z

def complex_to_cartesian(Z):
    x = np.real(Z).astype(np.float32)
    y = np.imag(Z).astype(np.float32)
    return x,y


#***********************************************************************
# Twisting functions
#**********************************************************************
class Escher:

    def __init__(self, coeff_):
        self.coeff = coeff_

    def log_z(self, Z):
        '''
        :param Z: grid of complex coordinates
        :return: log of the complex coordinates
        '''
        return np.log(Z)

    def rotation(self, Z):
        '''
        :param Z: grid of complex coordinates
        :param alpha: complex constant
        :return: rotated complex coordinates rotated by "alpha" degrees
        '''
        return Z* self.coeff

    def exp_z(self,Z):
        '''
        :param Z: grid of complex coordinates
        :return: exponential of the complex coordinates
        '''
        return np.exp(Z)

    def twist(self,Z):
        '''
        Function to receive straight grid of complex coordinates and warps them according to a coefficient
        Input: Takes complex coordinates, applies transform
        Output: Returns a single grid of complex coordinates
        '''
        #Log
        lnz = self.log_z(Z)
        #Rotation
        lnz_alpha = self.rotation(lnz)
        #Exponentiation
        ez = self.exp_z(lnz_alpha)
        return ez


#***********************************************************************
# Grid handling functions
#***********************************************************************

def meshgrid(src, resolution = 10):
    '''
    Function to meshgrid
    :param src:
    :return: meshgrid of X,Y coordinates in Cartesian plane
    '''
    src_row, src_col = src.shape[0], src.shape[1]
    out_row = src_row * resolution  # multiply by 20 to get higher resolution results
    out_col = src_row * resolution
    range_ = 1  # increasing this value from 1 puts more of the circle on the deformed output but also rotates it
    vec1 = np.linspace(-range_, range_, num=out_row, endpoint=True)
    vec2 = np.linspace(-range_, range_, num=out_col, endpoint=True)
    x, y = np.meshgrid(vec2, vec1)
    return x,y



def droste_transformation(x, y, c=1):
    for _ in range(2):
        #pixels out of bounds
        indx = (np.abs(x) >= c) | (np.abs(y) >= c)
        x[indx] *= 1. / droste_scale
        y[indx] *= 1. / droste_scale

        # Pixels close to zero
        indx = (np.abs(x) < c / droste_scale) & (np.abs(y) < c / droste_scale)
        x[indx] *= droste_scale
        y[indx] *= droste_scale
    return x, y



def out_file(x,y,src,outFile,c=1,resolution = 10):
    x,y = droste_transformation(x, y, c=1)
    #Center grid
    x2 = (x / c + 1) * src.shape[0] / 2
    y2 = (y / c + 1) * src.shape[1] / 2
    #Generate output image
    out2 = cv2.remap(src, x2, y2, interpolation=cv2.INTER_CUBIC, borderValue=200)
    cv2.imwrite(outFile, out2)


def main():

    # ****************************************************************
    # Input files
    # ****************************************************************
    inFile = 'images/escher_straight2.jpg'
    #outFile = 'images/circle_LOG_jim2.png'
    outFile = 'images/grid_straight2_try.png'

    #Image cropping
    #TODO: crop images (no margin)
    #TODO: make dimensions square (same hght and length)
    src = np.array(cv2.imread(inFile))

    # ****************************************************************
    # Escher Transformation
    # ****************************************************************

    #Create complex grid
    x, y = meshgrid(src) #center coordinates at (0,0)
    Z = cartesian_to_complex(x, y)  #convert Cartesian plane to complex


    # TODO: SELECT ONE: to do the forward deformation, use the reverse function (remap logic)
    #coeff = alpha      #From twisted to straight
    coeff = alpha_inv  #From straight to twisted

    # Apply Transformation
    escher = Escher(coeff)
    #Znew   = escher.twist(Z)

    #region
    #***************************************************************
    # Select for Individual operations (only for debugging)
    #*************************************************************

    #TODO: Don't forget to select the right coeff from above!!!
    #Znew = escher.log_z(Z)                 #Log
    #Znew = escher.rotation(Znew, coeff)    #Rotation
    Znew  = escher.exp_z(Z)              #Exponentiation
    #endregion


    # ****************************************************************
    # Display output picture
    # *****************************************************************
    X2, Y2 = complex_to_cartesian(Znew) # Back to Cartesians for display
    out_file(X2,Y2,src,outFile,c=1,resolution = 10) # Create output image




if __name__ == "__main__":
        main()

import numpy as np
import cv2
from PIL import Image
import logging
import matplotlib.pyplot as plt

# http://www.ams.org/notices/200304/fea-escher.pdf
#Math explained on min 20 here: https://www.youtube.com/watch?v=clQA6WhwCeA

#Interpolation and polar/ cart functions

deformation_scale = 256 #TODO: for Dmytro, this can be changed


#This one is not used
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#https://stackoverflow.com/questions/2164570/reprojecting-polar-to-cartesian-grid
#https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
#r = np.sqrt(x**2 + y**2)
#th = np.arctan2(y, x)
polar2z = lambda r,theta: r * np.exp( 1j * theta )
z2polar = lambda z: (abs(z), np.angle(z)) #awful rounding error from Python that complicates matters

#https://programmer.group/reading-notes-opencv-polar-coordinate-transformation.html
#r2, theta2 = cv2.cartToPolar(x, y, angleInDegrees=True)
#x, y = cv2.polarToCart(r1, theta, angleInDegrees=True) #theta is expressed in different units, see docum

def escher_deformation(x,y):
    '''
    Takes Cartesian coordinates, converts into polar, applies transform and returns cartesian outputs
    '''

    #FROM CARTESIAN TO COMPLEX
    Z = x+ y * 1j #== Z = r*np.exp(1j*th) #awful rounding error that complicates matters
    #r, th = z2polar(Z)  #complex in exponential representation -not used as it gives rounding errors

    #APPLY LOG TO THE COMPLEX PLANE
    #Z = np.log(r) + th * 1j #Log of complex nbr represented in exp form - small rounding errors on this one
    lnz = np.log(Z) # 1 line!

    #APPLY ESCHER TRANSFORM
    sn = 1 - (np.log(deformation_scale) * 1j * (1 / (2 * np.pi)))  #1- log(256)i/2.PI #TODO: Dmytro to change
    #Rotation
    lnz_sn = lnz * sn
    #Exponentiation
    ez = np.exp(lnz_sn)


    #DISPLAY - choose one to see output #TODO: JIM if you change this, you get the 3 steps
    #transform = lnz    #displays the log in the complex plane
    #transform = lnz_sn #displays the translated
    transform = ez       #displays the final exponential

    #OBS: if you do: transform = np.exp(lnz) you get back to the original picture, so this code is correct!

    #Back to Cartesians for display
    Xnew = np.real(transform)
    Ynew = np.imag(transform)

    return Xnew, Ynew


def create_X_new_and_Y_new(x, y, row, col):
    #TODO: Ask Jim what is going on here - this is for display but I am too dumb to understand the logic
    Xnew = (x / np.max(np.abs(x)) + 1) * col / 2
    Ynew = (y / np.max(np.abs(y)) + 1) * row / 2
    Xnew = np.clip(Xnew, 0, col - 1)
    Ynew = np.clip(Ynew, 0, row - 1)
    Xnew = np.floor(Xnew).astype(int)
    Ynew = np.floor(Ynew).astype(int)
    return Xnew, Ynew

def create_total_image_after_mapping(Xnew, Ynew, image):
    #TODO: JIM, this can be enhanced with your function, this is a dumb down version
        '''
        #Projects pixels of an image to the new coordinates
        :param Xnew: Xreal
        :param Ynew: Y-imaginary
        :return: Projected image
        '''
        row = image.shape[0]
        col = image.shape[1]
        new_img = np.zeros([row, col, 3], dtype=np.uint8) #TODO: this is for the 3 repetitions that I haven't coded
        new_img.fill(255) #white background
        #From old cartesian plane to new transformed plane
        for i in range(row):
            for j in range(col):
                new_img[Ynew[i][j], Xnew[i][j]] = image[i % col][j]#i % col #prevents loop from overlfowing
        return new_img


def main():

    inFile  = 'images/escher_straight2.jpg'
    outFile = 'images/escher_straight2_log_rotate_exp.png'

    #READ IMAGE
    src = np.array(cv2.imread(inFile))  # #TODO: Jim, this one reads JPG and PNG
    #src = np.stack([src[:m, :]] * 3, axis=2)  #stacking into 3RGB channels  #TODO: this was was giving me errors with different images, I am too lazy to improve it


    #CREATE CARTESIAN PLANE
    row, col = src.shape[0], src.shape[1]
    vec1 = np.linspace(-1, 1, num=row, endpoint=True)
    vec2 = np.linspace(-1, 1, num=col, endpoint=True)
    x,y = np.meshgrid(vec2, vec1)


    #APPLY DEFORMATION to the cartesian plane
    x, y = escher_deformation(x,y )


    #MAP IMAGE TO THE NEW CARTESIAN PLANE
    #TODO: Jim, no idea whatehell is this, prolly to go from -1,1 to Cartesian, need to review
    x, y = create_X_new_and_Y_new(x,y, row, col) #creates new plane, nto sure why, Jim knows
    img = create_total_image_after_mapping(x, y, src) #maps image to plane
    # out = cv2.remap(src, x, y, interpolation=cv2.INTER_CUBIC) #TODO: Jim's better solution,but I am too clumsy to make it work

    #Saves
    cv2.imwrite(outFile, img)


    #Tells me that it's done
    logging.warning('DONE')
    #Display
    cv2.imshow("transform", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
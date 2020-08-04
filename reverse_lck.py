import numpy as np
import cv2
from PIL import Image
import logging
import matplotlib.pyplot as plt


deformation_scale = 256 #TODO: for Dmytro, this can be changed


polar2z = lambda r,theta: r * np.exp( 1j * theta )
z2polar = lambda z: (abs(z), np.angle(z))

def escher_deformation(x,y):
    '''
    Takes Cartesian coordinates, converts into polar, applies transform and returns cartesian outputs
    '''

    #FROM CARTESIAN TO COMPLEX
    Z = x+ y * 1j

    #APPLY LOG TO THE COMPLEX PLANE
    lnz = np.log(Z)

    #APPLY ESCHER TRANSFORM
    sn = 1 - (2 * np.pi / np.log(deformation_scale) * 1j )

    #Rotation
    lnz_sn = lnz * sn

    #Exponentiation
    ez = np.exp(lnz_sn)


    #DISPLAY - choose one to see output
    #transform = lnz    #displays the log in the complex plane
    #transform = lnz_sn #displays the translated
    transform = ez       #displays the final exponential

    #OBS: if you do: transform = np.exp(lnz) you get back to the original picture, so this code is correct!

    #Back to Cartesians for display
    Xnew = np.real(transform)
    Ynew = np.imag(transform)

    return Xnew, Ynew


def create_X_new_and_Y_new(x, y, row, col):
    #Revert back to Cartesian plane
    Xnew = (x / np.max(np.abs(x)) + 1) * col / 2
    Ynew = (y / np.max(np.abs(y)) + 1) * row / 2
    Xnew = np.clip(Xnew, 0, col - 1)
    Ynew = np.clip(Ynew, 0, row - 1)
    Xnew = np.floor(Xnew).astype(int)
    Ynew = np.floor(Ynew).astype(int)
    return Xnew, Ynew

def create_total_image_after_mapping(Xnew, Ynew, image):
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

    inFile  = 'images/circle_log_rotate_exp.png'
    outFile = 'images/circle_log_rotate_exp_reverse.png'

    #READ IMAGE
    src = np.array(cv2.imread(inFile))
    #src = np.stack([src[:m, :]] * 3, axis=2)  #stacking into 3RGB channels

    #CREATE CARTESIAN PLANE
    row, col = src.shape[0], src.shape[1]
    vec1 = np.linspace(-1, 1, num=row, endpoint=True)
    vec2 = np.linspace(-1, 1, num=col, endpoint=True)
    x,y = np.meshgrid(vec2, vec1)


    #APPLY DEFORMATION to the cartesian plane
    x, y = escher_deformation(x,y )


    #MAP IMAGE TO THE NEW CARTESIAN PLANE
    x, y = create_X_new_and_Y_new(x,y, row, col) #creates new plane
    img = create_total_image_after_mapping(x, y, src) #maps image to plane
    # out = cv2.remap(src, x, y, interpolation=cv2.INTER_CUBIC) #TODO: implement Jim's better solution
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
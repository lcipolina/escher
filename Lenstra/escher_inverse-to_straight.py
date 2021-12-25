import math
import numpy as np
import cv2


#***********************************************************************
# Mapping functions
#***********************************************************************

def cartesian_to_complex(x, y):
    '''
    :param x: x-coordinates in Cartesian plane
    :param y: y-coordinates in Cartesian plane
    :return: one grid representing the complex plane
    '''
    Z = x + y * 1j
    return Z

def complex_to_cartesian(Z):
    x = np.real(Z)
    y = np.imag(Z)
    return x, y


#***********************************************************************
# Grid handling functions
#***********************************************************************

def meshgrid(src, resolution=10, limit=1):
    '''
    Function to meshgrid
    :param src: source image
    :param resolution: scaling of output image w.r.t. input image
    :param limit: scaling of the mapping
    :return: meshgrid of X,Y coordinates as complex values
    '''
    src_row, src_col = src.shape[0], src.shape[1]
    out_row = int(src_row * resolution)
    out_col = int(src_col * resolution)
    vec1 = np.linspace(-limit, limit, num=out_row, endpoint=True)
    vec2 = np.linspace(-limit, limit, num=out_col, endpoint=True)
    x, y = np.meshgrid(vec2, vec1)
    # Return pixel coordinates as complex values x+iy
    return cartesian_to_complex(x, y)


def select_solution(Z, limit=1):
    '''
    Select, for each pixel (x,y), one of the solutions Z(:,x,y) that is inside the limit
    :param Z: mapped coordinates, as complex values
    :param limit: limit of the coordinate system
    '''
    # Set pixels out of bounds to (0,0)
    indx = (np.abs(Z.real) > limit) | (np.abs(Z.imag) > limit)
    Z[indx] = 0
    # For pixels where multiple solutions are OK, select the one furthest
    # away from the origin
    out = Z[0]
    for ii in range(1, Z.shape[0]):
      tmp = Z[ii]
      indx = np.abs(tmp) > np.abs(out)
      out[indx] = tmp[indx]
    return out


def map_input(Z, src, limit=1):
    '''
    Function to create output image
    :param Z: mapped coordinates, as complex values
    :param src: source image
    :param limit: limit of the coordinate system
    '''
    # Convert complex values to coordinates
    x, y = complex_to_cartesian(Z)
    # Center grid (this converts coordinates [-limit,limit] to [0,src.shape])
    x = (x / limit + 1) * src.shape[0] / 2
    y = (y / limit + 1) * src.shape[1] / 2
    # Generate output image
    return cv2.remap(src, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_CUBIC, borderValue=200)


def escher_inverse(src, zoom=1, resolution=10, counter_rotation=0):
    '''
    Main function: applies the inverse Escher mapping
    :param src: image to apply the mapping to
    :param zoom: zooming of the ouput image
    :resolution: how many pixels compared to src
    :param counter_rotation: additional rotation given to the image to align visual axis
    '''
    # Create complex grid
    Z = meshgrid(src, resolution, zoom) #center coordinates at (0,0)
    # Apply Inverse Transformation
    Z = np.log(Z)
    Z = np.stack((Z + 2j * math.pi, Z, Z - 2j * math.pi, Z - 4j * math.pi), 0) # mutliple solutions to log(Z)
    Z /= (2j * math.pi - math.log(256)) / (2j * math.pi)
    Z = np.exp(Z)
    # Apply an additional rotation
    Z *= math.cos(counter_rotation) + 1j * math.sin(counter_rotation)
    # Select among the solutions for each pixel the best one
    Z = select_solution(Z, 1)
    # Create output picture
    return map_input(Z, src, 1)


#***********************************************************************
# Driver
#***********************************************************************

if __name__ == "__main__":
   # Parameters of the mapping
   resolution = 3    # larger number means more output pixels - it takes longer but increases the resol and img size

   # Image data to map
   src = cv2.imread('escher_scanned_poster_cropped.jpg', cv2.IMREAD_GRAYSCALE) #Better use an image already trimmed.. it impacts a lot the results
   #src = src[97:723, 104:744]
   #src = src[90:715, 104:741]     # this one makes the output a bit more straight

   for zoom in range(7):  #'zoom' parameter controls how much inward in the center we want to go - we need at least 6 outputs to get inside the painting (into the gallery)
      # Generate mapped image
      rot_angle = 0.65  #experimental (to make output straight)
      out = escher_inverse(src, 2**(-zoom), resolution, rot_angle) # rotation angle 0.65 determined experimentally
      # Write to file
      print('here!!!')
      cv2.imwrite('escher_straightened_'+str(zoom)+'.jpg', out)

   print('done!!!')
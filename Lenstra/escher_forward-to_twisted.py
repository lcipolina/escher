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


def map_input(Z, src, src_scale, droste_scale):
    '''
    Function to create output image
    :param Z: mapped coordinates, as complex values
    :param src: images to apply the mapping to
    :param src_scale: scale for each of the images
    :param droste_scale: the scale at which we get back to the first image
    '''
    assert len(src) == len(src_scale)
    src_size = src[0].shape
    for ii in range(1, len(src)):
        assert src[ii].shape == src_size
    # Convert complex values to coordinates
    x, y = complex_to_cartesian(Z)
    # Create a map for which pixels to take from which input image
    unused = np.full(x.shape, True)
    out = np.full(x.shape, 0)
    for ii in reversed(range(len(src))):
        x0 = ((x * src_scale[ii] + 1) * (src_size[0] / 2)).astype(np.float32)
        y0 = ((y * src_scale[ii] + 1) * (src_size[1] / 2)).astype(np.float32)
        tmp = cv2.remap(src[ii].astype(float), x0, y0, interpolation=cv2.INTER_NEAREST, borderValue=-1)
        mask = (tmp >= 0) & unused
        out[mask] = tmp[mask]
        unused ^= mask
    print("Pixels not assigned:", np.count_nonzero(unused))
    out = out.astype(np.uint8)
    return out


def escher(src, src_scale, zoom=1, resolution=10, counter_rotation=0):
    '''
    Main function: applies the Escher mapping
    :param src: images to apply the mapping to
    :param src_scale: scale for each of the images
    :param zoom: zooming of the ouput image, by default is 1
    :resolution: how many pixels compared to src
    :param counter_rotation: additional rotation given to the image to align visual axis
    '''
    # Create complex grid
    Z = meshgrid(src[0], resolution, zoom) #center coordinates at (0,0)
    # Apply Transformation
    Z = np.log(Z)
    Z = np.stack((Z + 2j * math.pi, Z, Z - 2j * math.pi, Z - 4j * math.pi), 0) # mutliple solutions to log(Z)
    Z *= (2j * math.pi - math.log(256)) / (2j * math.pi)
    Z = np.exp(Z)
    # Apply an additional rotation
    Z *= math.cos(counter_rotation) + 1j * math.sin(counter_rotation)
    # Select among the solutions for each pixel the best one
    Z = select_solution(Z, 1)
    # Create output picture
    return map_input(Z, src, src_scale, 256)


#***********************************************************************
# Driver
#***********************************************************************

if __name__ == "__main__":
   # Parameters of the mapping
   resolution = 1                 # larger number means more output pixels
   zoom = 1                      # this value can be used to zoom in our out of the generated image

   # Image data to map
   src = []
   src_scale = []
   for ii in range(8):
      src.append(cv2.imread('escher_straightened_'+str(ii)+'.jpg', cv2.IMREAD_GRAYSCALE))
      src_scale.append(2**ii)     # doesn't match values used in escher_inverse.py
   
   # Generate mapped image
   rot_angle = 0 #0.65  # experimental
   out = escher(src, src_scale, zoom, resolution, rot_angle) # rotation angle copied from escher_inverse.py
   
   # Write to file
   cv2.imwrite('escher_composite-lck.jpg', out)
   print('done!!')
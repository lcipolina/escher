import math
import diplib as dip
import numpy as np


#***********************************************************************
# Grid handling functions
#***********************************************************************

def meshgrid(src, resolution=2, limit=1):
    '''
    Function to meshgrid
    :param src: source image
    :param resolution: scaling of output image w.r.t. input image
    :param limit: scaling of the mapping
    :return: meshgrid of X,Y coordinates as complex values
    '''
    out_sizes = [int(src.Size(0) * resolution), int(src.Size(1) * resolution)]
    out = dip.CreateCoordinates(out_sizes, {"frequency"}) * (2 * limit)
    out.MergeTensorToComplex()
    return out


def select_solution(Z, limit=1):
    '''
    Select, for each pixel (x,y), one of the solutions Z(:,x,y) that is inside the limit
    :param Z: mapped coordinates, as complex values
    :param limit: limit of the coordinate system
    '''
    # Set pixels out of bounds to (0,0)
    indx = (dip.Abs(Z.Real()) > (limit * 0.998)) | (dip.Abs(Z.Imaginary()) > (limit * 0.998))
    Z[indx] = 0
    # For pixels where multiple solutions are OK, select the one furthest
    # away from the origin
    out = Z(0)
    for ii in range(1, Z.TensorElements()):
      tmp = Z(ii)
      indx = dip.Abs(tmp) > dip.Abs(out)
      if dip.Any(indx)[0][0]:
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
    src_size = src[0].Sizes()
    for ii in range(1, len(src)):
        assert src[ii].Sizes() == src_size
    # Convert complex values to coordinates
    Z.SplitComplexToTensor()
    # Create a map for which pixels to take from which input image
    out = dip.Image(Z(0).Sizes(), src[0].TensorElements(), 'SFLOAT')
    out.SetColorSpace(src[0].ColorSpace())
    out.Fill(-1)
    for ii in reversed(range(len(src))):
        Z0 = dip.MultiplySampleWise(Z + (1 / src_scale[ii]), dip.Create0D([src_size[0] * src_scale[ii] / 2, src_size[1] * src_scale[ii] / 2]))
        src[ii].Convert('SFLOAT')
        tmp = dip.ResampleAt(src[ii], Z0, method="linear", fill=-1)  # method="3-cubic" for better quality, but I don't think it matters
        mask = (tmp(0) >= 0) & (out(0) < 0)
        if dip.Any(mask)[0][0]:
           out[mask] = tmp[mask]
    print("Number of pixels not assigned:", dip.Count(out(0) < 0))
    out.Convert("UINT8")
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
    Z = meshgrid(src[0], resolution, zoom)
    # Apply Transformation
    Z = dip.Ln(Z)
    Z = dip.JoinChannels((Z + 2j * math.pi, Z, Z - 2j * math.pi, Z - 4j * math.pi)) # mutliple solutions to log(Z)
    Z *= (2j * math.pi - math.log(256)) / (2j * math.pi)
    Z = dip.Exp(Z)
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
   resolution = 1                # larger number means more output pixels
   zoom = 1                      # this value can be used to zoom in our out of the generated image

   # Image data to map
   src = []
   src_scale = []
   for ii in range(8):
      src.append(dip.ImageRead('escher_straightened_'+str(ii)+'.jpg'))
      src_scale.append(2**ii)     # doesn't match values used in escher_inverse.py
   
   # Generate mapped image
   out = escher(src, src_scale, zoom, resolution)
   
   # Write to file
   dip.ImageWrite(out, 'escher_composite.jpg')

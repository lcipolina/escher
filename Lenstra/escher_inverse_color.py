import math
import diplib as dip


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
    indx = (dip.Abs(Z.Real()) > limit) | (dip.Abs(Z.Imaginary()) > limit)
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


def map_input(Z, src, limit=1):
    '''
    Function to create output image
    :param Z: mapped coordinates, as complex values
    :param src: source image
    :param limit: limit of the coordinate system
    '''
    # Convert complex values to coordinates
    Z.SplitComplexToTensor()
    # Center grid (this converts coordinates [-limit,limit] to [0,src.shape])
    Z += limit
    Z = dip.MultiplySampleWise(Z, [src.Size(0) / (2 * limit), src.Size(1) / (2 * limit)])
    # Generate output image
    out = dip.ResampleAt(src, Z, method="linear", fill=200)  # method="3-cubic" for better quality, but I don't think it matters
    out.SetColorSpace(src.ColorSpace())  # why is this necessary?
    return out


def escher_inverse(src, zoom=1, resolution=2, counter_rotation=0):
    '''
    Main function: applies the inverse Escher mapping
    :param src: image to apply the mapping to
    :param zoom: zooming of the ouput image
    :resolution: how many pixels compared to src
    :param counter_rotation: additional rotation given to the image to align visual axis
    '''
    # Create complex grid
    Z = meshgrid(src, resolution, zoom)
    # Apply Inverse Transformation
    Z = dip.Ln(Z)
    Z = dip.JoinChannels((Z + 2j * math.pi, Z, Z - 2j * math.pi, Z - 4j * math.pi)) # mutliple solutions to log(Z)
    Z /= (2j * math.pi - math.log(256)) / (2j * math.pi)
    Z = dip.Exp(Z)
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
   resolution = 2                 # larger number means more output pixels

   # Image data to map
   src = dip.ImageRead('1.PNG')
   # Picking the right cropping is important, as it determines the origin
   src = src[104:741, 90:715]

   for zoom in range(9):
      # Generate mapped image
      out = escher_inverse(src, 2**(-zoom), resolution)
      # Write to file
      dip.ImageWrite(out, 'escher_straightened_'+str(zoom)+'.jpg')

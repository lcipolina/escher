import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cosine
import imutils

errors = []
optimization_history_cost = []
optimization_history_params = []

def load_images():
    def open_image(name: str):
        img = cv2.imread(name)
        return img
    base='../images/centered/'
    names = ['1.png', '2.png', '3.png','4.png','5.png']
    response = {i+1:open_image(base+n) for i,n in enumerate(names)}
    return response

def view(l: list, deg:float, scaling:float):
    f, ax = plt.subplots(1,len(l),figsize=(5*len(l),5))
    for j,img in enumerate(l):
      ax[j].imshow(img, cmap='gray_r')
    if len(l)==3:
      ax[0].set_title('base')
      ax[1].set_title(f'deg: {deg}\nscaling: {scaling}')
      ax[2].set_title('target')

    plt.savefig('view.png')

def borders(imgs):
    def extract_border(img):
        return cv2.Canny(img, 200, 255)
    result =  {1:extract_border(imgs[1]),
            2:extract_border(imgs[2]),
                }
    result =  {k:np.asarray(v, dtype=float) for k,v in result.items()}
    #result = {k: np.where(v.flatten()>0,255,0).reshape(v.shape).astype(float) for k,v in result.items()}
    return result

def subtract_images(imgs):
    a,b,c = [np.asarray(imgs[i], dtype=float) for i in [1,2,3]]
    response = {1:a-b, 
                2:b-c,
                }
    return response

def set_the_context(img1, img2):
    original_width = img1.shape[0]
    original_height = img1.shape[1]
    def rotate(img, deg):
        return imutils.rotate(img, angle=-1*deg)
    def scale(img, factor):
        if factor>1: return img
        base_image = np.zeros(img.shape)
        new_image = imutils.resize(img, width=int(original_width*factor))
        a,b,c,d, =       ( new_image.shape[0]//2, 
                            new_image.shape[0]-new_image.shape[0]//2,
                           new_image.shape[1]//2,
                              new_image.shape[1]-new_image.shape[1]//2,
                         )

        base_image[original_width//2-a:original_width//2+b, 
                original_height//2-c:original_height//2+d,:] = new_image
        return base_image

    def scale(img, f: float):
      """
      scales by a factor "f"
      """
      # Store the original shape
      s = img.shape
      width = int(img.shape[1] * f)
      height = int(img.shape[0] * f)
      dim = (width, height)

      if f>1:
        # Define the pixel indexes in order to
        # return an image with equal amount of pixels
        ind = [
               slice(
                   int((s[0]*f)//2-s[0]//2),
                   int((s[0]*f)//2+s[0]//2),
               ),
               slice(
                   int((s[1]*f)//2-s[1]//2),
                   int((s[1]*f)//2+s[1]//2),
               ),
        ]
        if len(s)>2:
          return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)[ind[0],ind[1],:]
        else:
          return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)[ind[0],ind[1]]
      canvas = np.zeros(s)
      new = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
      a,b,c,d = s[0]//2-height//2, s[0]//2+height//2, s[1]//2-width//2, s[1]//2+width//2
      if len(s)>2:
        canvas[a:b,c:d,:] = new[:b-a, :d-c,:]
      else:
        canvas[a:b,c:d] = new[:b-a, :d-c]
      return canvas

    return img1, img2, rotate, scale

def set_the_cost_function(img):
    v_old = np.asarray(img).flatten()
    def cost(v_new):
        result = 1-cosine(v_old, np.asarray(v_new).flatten())
        optimization_history_cost.append(result)
        return result
    return cost

def set_the_mapping(img, rotate, scale):
    def mapping(d, s):
        optimization_history_params.append((d,s))
        return scale(rotate(img, d), s)
    return mapping

def optimize_with_history(f, d, s, optimizer='Nelder-Mead'):
    print(f'el resultado fue: {minimize(f,(d,s), method=optimizer)}')
    return list(zip(*optimization_history_params)), optimization_history_cost

def plot_the_history(history, which=[1,2,3]):
    print('history is ',history)
    f,ax = plt.subplots(figsize=(15,7))
    if 1 in which: ax.plot(history[0][0], label='degrees')
    if 2 in which: ax.plot(history[0][1], label='scaling')
    if 3 in which: ax.plot(history[1], label='cost')
    ax.legend()
    ax.grid()
    plt.show()

def get_cost_vs_degrees(cost, mapping, factor, smaller_degree_range=np.linspace(-180,180,180)):
    results = []
    for deg in smaller_degree_range:
        print(f'deg: {deg}')
        results.append(cost(mapping(deg, factor)))
    return [results, smaller_degree_range]

def plot_cost_vs_degrees(values):
    plt.plot(values)
    plt.title('cost function vs degrees')
    plt.show()

def main():
    # LOAD
    original_imgs = load_images()
    imgs = original_imgs
   
    # CHOOSE INITIAL PARAMETERS AND SET THE INDEXES
    import sys
    try:
        initial_degrees, initial_scaling = float(sys.argv[1]), float(sys.argv[2])
        IX1, IX2 = int(sys.argv[3]),int(sys.argv[4])
    except:
        initial_degrees, initial_scaling = 20,0.8
        IX1, IX2 = 4,5

    # SET THE CONTEXT
    img1, img2, rotate, scale = set_the_context(imgs[IX1],imgs[IX2])

    #plt.imshow(img2);plt.show()
    #a,b = img2.shape[:2]
    #newa,newb=2*a,2*b
    #plt.imshow(np.asarray(cv2.resize(img2, (newa,newb), cv2.INTER_CUBIC))[newa//2-a//2:newa//2+a//2, newb//2-b//2:newb//2+b//2]);plt.show()
    #img1 = scale(img1,3)
    #img2 = scale(img2,3)



    # SET THE COST FUNCTION
    cost = set_the_cost_function(img2)

    # SET THE MAPPING FROM INPUT TO OUTPUT
    mapping = set_the_mapping(img1, rotate, scale)

    # PLOT THE COST FUNCTION VS DEGREE IN GENERAL
    if True:
        TEMPORAL = {}
        for s in np.linspace(0.2,1,10):
            best_scaling_so_far = s#0.583
            cost_vs_degrees = get_cost_vs_degrees(cost, mapping, best_scaling_so_far,  smaller_degree_range=np.linspace(-90,90,10))
            TEMPORAL[s] = cost_vs_degrees.copy()
        with open('RESULTS.pkl','wb') as f:
                import pickle
                pickle.dump(TEMPORAL, f)



    # BUILD THE FUNCTION TO MINIMIZE
    function_to_minimize = lambda params: cost(mapping(params[0], params[1]))

    # PERFORM THE OPTIMIZATION AND RETRIEVE THE HISTORY
    history = optimize_with_history(function_to_minimize,
                                    initial_degrees,
                                    initial_scaling,
                                    optimizer='Nelder-Mead')

    # SAVE THE HISTORY
    import pickle
    try:
      with open('data.pkl','rb') as f:
          data = pickle.load(f)
    except:
      data={}
    data[(initial_degrees, initial_scaling)] = history
    with open('data.pkl','wb') as f:
        pickle.dump(data, f)

    # Plot the result
    deg, scaling = data[(initial_degrees, initial_scaling)][0][0][-1],data[(initial_degrees, initial_scaling)][0][1][-1]
    view([img1, scale(rotate(img1,deg),scaling), img2], deg, scaling)

    # Plot the cost function close to the relevant zone please
    if True:
        degr = np.linspace(data[(initial_degrees, initial_scaling)][0][0][-1]-2,data[(initial_degrees, initial_scaling)][0][0][-1]+2,5)
        for s in np.linspace(max(data[(initial_degrees, initial_scaling)][0][1][-1]-0.1,0.1),data[(initial_degrees, initial_scaling)][0][1][-1]+0.1,5):
            best_scaling_so_far = s#0.583
            cost_vs_degrees = get_cost_vs_degrees(cost, mapping, best_scaling_so_far, smaller_degree_range=degr)
            TEMPORAL[s] = cost_vs_degrees.copy()
        with open('RESULTSdetail.pkl','wb') as f:
                import pickle
                pickle.dump(TEMPORAL, f)


    from plotcost import plot_optimization_path
    plot_optimization_path()

if __name__=='__main__':
    print('about to run!')
    main()
    print(f'ended with {len(errors)} errors')

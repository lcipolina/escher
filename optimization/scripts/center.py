import cv2
import numpy as np
import sys

# read image as grayscale
#img = cv2.imread(sys.argv[1], cv2.COLOR_BGR2GRAY)
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
print(img.shape)

# get shape
hh, ww = img.shape[:2]


# get contours (presumably just one around the nonzero pixels) 
contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    x,y,w,h = cv2.boundingRect(cntr)

# recenter
startx = (ww - w)//2
starty = (hh - h)//2
result = np.zeros_like(img)
result[starty:starty+h,startx:startx+w] = img[y:y+h,x:x+w]

# view result
#cv2.imshow("RESULT", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# save reentered image
cv2.imwrite('centered-'+sys.argv[1],result)


def r(r,img1,f):
  a,b = img1.shape[:2]
  newa,newb = int(f*a),int(f*b)
  return rotate(np.asarray(cv2.resize(img1, (newa,newb), cv2.INTER_CUBIC))[newa//2-a//2:newa//2+a//2, newb//2-b//2:newb//2+b//2],-1*r)

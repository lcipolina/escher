import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from imutils import rotate
import cv2
from imageio import imread

ROTATION = 162.497433
SCALING = 0.05576523
l=['','w','b'][0]

try :mask = np.asarray(Image.open('crop1mask.jpg')).astype('float32')
except: mask = np.asarray(Image.open('crop1mask.png')).astype('float32')
y = np.asarray(Image.open(f'escher{l}.jpg')).astype('float32')
ypred = np.asarray(Image.open(f'ypred{l}.jpg')).astype('float32')
full = np.asarray(Image.open('image.png')).astype('float32')


assert (ypred.shape==y.shape)
assert (ypred.shape[0]==mask.shape[0])
assert (ypred.shape[1]==mask.shape[1])


for i in range(mask.shape[0]-1):
	if (mask[0,i,0]==255 and mask[0,i+1,0]==0):
		IX = i+1
		break


# Y base
y_cropped = y[:,:IX,:]
pad=512-IX#int(2*2*(0.03473422001351492* y_cropped.shape[0])//2)#512-IX #
y_cropped = np.concatenate([np.zeros((y_cropped.shape[0], pad, 3)), y_cropped],1)



# Predictions
ypred_cropped = np.concatenate([np.zeros((ypred.shape[0],IX,3)), ypred[:,IX:,:]],1)



# Enlarge Y base by a factor SCALING^-1
Lx=y_cropped.shape[0]
Ly=y_cropped.shape[1]
Lxbig = int(Lx*18)
Lybig = int(Ly*18)
sq = 1


# print gallery with some columns of 0s on the left, and HUGE
y_new = cv2.resize(y_cropped[Lx//2-int(sq*Lx//2):Lx//2+int(sq*Lx//2),
			     Ly//2-int(sq*Ly//2):Ly//2+int(sq*Ly//2)], 
			    (int(sq*Lxbig//2)*2,
			    int(sq*Lybig//2)*2), cv2.INTER_CUBIC)


# Predictions embedded in a HUGE background
ypred_new = np.zeros(y_new.shape)
P=pad
ypred_new[ypred_new.shape[0]//2-ypred_cropped.shape[0]//2:ypred_cropped.shape[0]//2+ypred_new.shape[0]//2,
		 ypred_new.shape[1]//2-ypred_cropped.shape[1]//2+P:ypred_cropped.shape[1]//2+ypred_new.shape[1]//2+P, :] = ypred_cropped



# Escher print gallery embedded in a HUGE background
ycropped_new = np.zeros(y_new.shape)
ycropped_new[ycropped_new.shape[0]//2-y_cropped.shape[0]//2:y_cropped.shape[0]//2+ycropped_new.shape[0]//2,
		 ycropped_new.shape[1]//2-y_cropped.shape[1]//2:y_cropped.shape[1]//2+ycropped_new.shape[1]//2, :] = (
			y_cropped[:2*(y_cropped.shape[0]//2), :2*(y_cropped.shape[1]//2), :])




# Sum both small images (print gallery + prediction)
#ypred_new = rotate(ypred_new, -1*ROTATION)
#total = (rotate(ycropped_new, -1*ROTATION)+
#				ypred_new).astype('uint8')
total = ycropped_new+ypred_new
total = (rotate(total,-1*ROTATION))
fixer = np.zeros(total.shape)
a=2341-2299 #0.018703784254023487
b=abs(2224-2302) #0.03388357949609035
#a = int(0.018703784254023487*fixer.shape[0])
#b = int(0.03388357949609035*fixer.shape[1])
a*=2
b//=2
fixer[:-a,b:,:] = total[a:,:-b,:]
total=fixer.astype('uint8')


base = y_new.copy().astype('uint8')
assert (base.shape==total.shape)
result = np.where(total.flatten()!=0, total.flatten(), base.flatten()).reshape(base.shape)[:,pad*18:,:]



if False:
	f, ax = plt.subplots(3,3,figsize=(25,25))
	ax[0,0].imshow(y_cropped.astype('uint8'))
	ax[0,1].imshow(ypred_cropped.astype('uint8'))
	ax[0,2].imshow(full.astype('uint8'))
	ax[1,0].imshow((y_new).astype('uint8'))
	ax[1,1].imshow((ypred_new).astype('uint8'))
	ax[1,2].imshow(rotate(ycropped_new, -1*ROTATION).astype('uint8'))
	ax[2,0].imshow(total.astype('uint8'))
	ax[2,1].imshow(result)
	for _1 in range(3):
		for _2 in range(3):
			ax[_1,_2].axis('off')
	plt.show()
else:
	f, ax = plt.subplots(1,1,figsize=(20,20))
	ax.imshow(result)
	ax.axis('off')
	plt.savefig('somth.png')
	im = imread('somth.png', pilmode="L")
	f, ax = plt.subplots(1,1,figsize=(20,20))
	ax.imshow(im, cmap='gray')
	ax.axis('off')
	plt.savefig('somth.png')
	#plt.show()











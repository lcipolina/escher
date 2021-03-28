from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

def report(name):
	img = Image.open(name)
	imgn = np.asarray(img,dtype='float')
	imgn = imgn.reshape(imgn.shape[0]*imgn.shape[1],-1)
	print(imgn.shape)
	mu, std = np.mean(imgn,0), np.std(imgn,0)
	print(f'\nmean: {mu}, std: {std}\n')
	return mu,std
	
def give_some_new_values(name, mu, extstd):
	img = Image.open(name)
	imgn = np.asarray(img,dtype='float')
	s=imgn.shape
	if s[0]*s[1]>3e6:
		print('size was...'+str(s)) 
		raise Exception('too big!')
	print(s)
	imgn = imgn.reshape(s[0]*s[1],-1)
	if True:
		return np.mean(imgn.reshape(s[0],s[1],-1),2).reshape(s[0],s[1],1)
	vn=(np.sqrt(sum(mu**2)))**2
	#print('VN IS ',vn)
	#c = np.matmul(np.matmul(imgn,mu).reshape(-1,1), mu.reshape(1,3))/vn
	print(np.matmul(imgn,mu).reshape(-1,1).shape,'MATMUL WAS')
	#plt.hist(np.matmul(imgn,mu).flatten());plt.show()
	c = np.matmul(np.matmul(imgn,mu).reshape(-1,1), mu.reshape(1,3))/vn
	c=c.reshape(*s)
#	for i in range(len(c.shape[0])):
#		for j in range(len(c.shape[1])):
#			if c[i,j,0]
#	c = np.matmul(np.matmul(imgn,mu).reshape(-1,1), mu.reshape(1,3))/vn
#	c=c.reshape(*s)
	
	if False:
		plt.hist(c[:,:,0].flatten(),label='one',alpha=0.5)
		plt.hist(c[:,:,1].flatten(),label='two',alpha=0.5)
		plt.hist(c[:,:,2].flatten(),label='three',alpha=0.5)
		plt.legend()
	elif False:
		index = range(len(c[:,:,0].flatten()))
		ix = np.random.choice(index,10000)
		if False:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(c[:,:,0].flatten()[ix],c[:,:,1].flatten()[ix],c[:,:,2].flatten()[ix])
			ax.set_xlabel('r')
			ax.set_ylabel('g')
		else:
			plt.scatter(c[:,:,1].flatten()[ix],c[:,:,2].flatten()[ix])
		
	elif False:
		f, ax = plt.subplots(1,3,figsize=(20,7))
		for i in range(3): ax[i].imshow(c[:,:,i], cmap='gray')
	elif False:
		plt.imshow(np.abs(c).astype('uint8'))
	plt.show()
	#c = np.where(c.flatten()<0,0,c.flatten()).reshape(c.shape).astype('int')
	return c.astype('uint8')


#mu1,std1 = report(sys.argv[1])

mu1 = np.asarray([181.87769166, 173.34503561, 163.64049016])
std1 = [65.35000492, 65.35622675, 63.92069093]

L=0
errors=[]
errors2=[]
try:
	os.mkdir('../news')
except:
	print('dir news already existed')
for d in [x for x in os.listdir() if x[-3:]!='.py' and x[:3]!='NEW']:
	try:
		os.mkdir(f'../news/NEW-{d}')
	except Exception as ins:
		errors.append(ins.args)
	for f in os.listdir(d):
		nm = f'{d}/{f}'
		nm_out = f'../news/NEW-{d}/{f}'  
		try:
			newimg = give_some_new_values(nm,mu1,std1)
			#print(newimg.shape, 'after')
			#img = Image.fromarray(newimg, mode='rgb')
			#print('FINALLY:',img.size)
			cv2.imwrite(nm_out, newimg)#cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR))    
			#img.save(nm_out)
			#del(newimg)
		except Exception as ins:
			errors2.append(ins.args)
			print(f'{nm} failed!')
			L+=1
                    
print(f'\n\n{L} FAILS\n\n')

print(errors2)




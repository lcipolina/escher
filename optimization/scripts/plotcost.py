import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import pickle
import matplotlib.pyplot as plt
import random
import sys


def plot_optimization_path():
	with open('RESULTS.pkl','rb') as f:
	    data = pickle.load(f)
	keys = list(data.keys())
	values= [data[k] for k in keys]
	scale = keys
	cost = [x[0] for x in values]
	degrees = [x[1] for x in values][0]

	#print(f'\nscale is {scale}\ndegrees are {degrees}\nand cost is {cost}')

	np.save('degrees', np.asarray(degrees))
	np.save('scale', np.asarray(scale))
	np.save('cost', np.asarray(cost))

	with open('data.pkl','rb') as f:
	    data = pickle.load(f)

	for k in data.keys():
	    data[k] = [data[k][0][0],data[k][0][1], data[k][1]]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d',)# figsize=(10,10))
	x = np.load('degrees.npy')
	y = np.load('scale.npy')
	X, Y = np.meshgrid(x, y)
	Z = np.load('cost.npy')

	ax.plot_surface(X, Y, Z, alpha=0.35)

	ax.set_xlabel('DEGREES')
	ax.set_ylabel('SCALING')
	ax.set_zlabel('COSINE DISTANCE COST')

	color={0:'red',1:'green',2:'yellow'}
	i=0
	print(data.keys())
	for k in data.keys():
	    L = len(data[k][0])
	    ix = sorted(np.random.choice(range(L//3), int(L*0.2)).tolist()) + sorted(np.random.choice(range(2*L//3,L), int(L*1.5)).tolist())
	    p=0.25
	    ix = [i*int(L*p) for i in range(L//int(L*p))]
	    #print(ix)
	    ax.plot(*[np.asarray(X)[ix] for X in data[k]], label=str(k), c=color[i],lw=2)
	    #for i in range(3): print(data[k][i][-5:])
	    i+=1

	ax.legend()

	#ax.set_xticklabels()

	#ax.view_init(30, angle)

	plt.savefig('optimization-path.png')

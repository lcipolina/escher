import os, time, shutil

import sys

# make input directory
try:
	os.system('rm -r input ; mkdir input')
except: 
	pass

BASE_DIR = 'original_edges/all_together'

COUNTER, BATCH_SIZE = 0, 2
for nm in [x for x in os.listdir(BASE_DIR) if x[-8:-4]=='edge']:
	nm1 = nm
	nm2 = nm[:-9]+'_gf.jpg'
	shutil.copy2(f'{BASE_DIR}/'+nm1,'input/'+nm1)
	shutil.copy2(f'{BASE_DIR}/'+nm2,'input/'+nm2)
	COUNTER+=1
	if COUNTER==BATCH_SIZE:
		try:
			os.system(f'python3 test.py  --outline_style {sys.argv[1]}  --shading_style {sys.argv[2]}')
		except:
			os.system('python3 test.py')
		COUNTER = 0 #LUCIA
		os.system('rm input/*')


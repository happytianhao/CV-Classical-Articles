import os

for str in os.listdir()[1:-2]:
	print('####',str.split('.')[0])
import csv
import numpy as np

def import_csv(filename,flag = True):
	data = []
	with open(filename) as datafile:
		csvReader = csv.reader(datafile)
		for row in csvReader:
			if(flag):
				picture = row[1].split()
				data.append([int(picture[i]) for i in range(0,len(picture))])
			else:
				data.append(int(row[0]))
	return data


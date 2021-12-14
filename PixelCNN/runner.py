import os
import subprocess

# set the values for size of kernel and size of matrix
k = [23,33]
hw = [150]

#iterate over kernel size and matrix size
for i in k:
	for j in hw:
		sum = 0
		# delete the previous version of out.txt
		os.remove("out.txt")
		print("for kernel",i,"width",j)

		# run the code 4 times and store the value in out.txt
		for n in range(4):
			os.system(f'./st1.o -k {i} -hw {j} >> out.txt')
			
		# open out.txt to get the values, take average, and print out
		f = open("out.txt")
		fl = f.readlines()
		for line in fl:
			l = line.split()
			print(l[0])
			sum = sum + float(l[0])
		print(sum/4, int(l[1]), int(l[2]))
			

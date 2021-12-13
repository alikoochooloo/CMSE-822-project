import os
import subprocess

k = [23,33]
hw = [150]
for i in k:
	for j in hw:
		sum = 0;
		os.remove("out.txt")
		print("for kernel",i,"width",j)
		for n in range(4):
			#print("run", n)
			os.system(f'./st1.o -k {i} -hw {j} >> out.txt')
			#proc = subprocess.Popen(['/saffarym/Desktop/822/project',f'./normal.o -k {i} -hw {j}'])
			#(out,err) = proc.communicate()
			#print(out)
		f = open("out.txt")
		fl = f.readlines()
		for line in fl:
			l = line.split()
			print(l[0])
			sum = sum + float(l[0])
		print(sum/4, int(l[1]), int(l[2]))
			

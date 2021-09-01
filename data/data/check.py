import os


for root,dirs,files in os.walk("./"):
	if len(dirs)>0:
		for dir in dirs:
			print(dir)
			for _,__,___ in os.walk("./"+dir):
				print(len(___))
			

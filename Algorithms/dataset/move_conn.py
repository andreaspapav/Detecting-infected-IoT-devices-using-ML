import os
import shutil

directory = r'/cs/student/projects1/2017/apapavas/Dissertation/Algorithms/dataset/opt/Malware-Project/BigDataset/IoTScenarios'

counter = 1

for filename in os.listdir(directory):
	print(filename)
	file_new = "/cs/student/projects1/2017/apapavas/Dissertation/Algorithms/dataset/opt/Malware-Project/BigDataset/IoTScenarios/"+ filename + "/bro/conn.log.labeled"
	file_tar = "/cs/student/projects1/2017/apapavas/Dissertation/Algorithms/dataset/original_dataset/conn" + str(counter) + ".log.labeled"
	shutil.move(file_new,file_tar)
	counter +=1



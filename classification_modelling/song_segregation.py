import pickle
import os
import shutil

folder_path = input("Enter folder where songs to be segregated are present : ")
result_file = input("Enter file name to load the predictions results for segregation : ")

with open("artefacts" + "/" + result_file, "rb") as result_file:
    result_list = pickle.load(result_file)

os.mkdir(folder_path + "/" + "happy")
os.mkdir(folder_path + "/" + "sad")
os.mkdir(folder_path + "/" + "thriller")

for item in result_list:
    file_name = str(item[0]).replace("-output.wav", "").replace("0_", "")
    if item[1] == 0:
        shutil.move(folder_path + "/" + file_name, folder_path + "/" + "happy")
    elif item[1] == 1:
        shutil.move(folder_path + "/" + file_name, folder_path + "/" + "sad")
    elif item[1] == 2:
        shutil.move(folder_path + "/" + file_name, folder_path + "/" + "thriller")



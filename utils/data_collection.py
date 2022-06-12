from ast import arg
from importlib.resources import read_text
import string
import youtube_dl
import csv
import sys,os


def read_txt(f_path):
    with open(f_path,'r') as file:
        lines = file.readlines()
    l_l=[]
    for l in lines:
        l_l.append([l])
    return l_l


def read_csv(f_path):
    """
    Function to read url from csv file
    :return: a list of urls 
    """
    lists_from_csv = []
    with open(f_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        if header != None:
            for row in csv_reader:
                lists_from_csv.append(row)
    return lists_from_csv

def run(songs_list):
    
    """
    General function to convert youtube vido to mp3 and saves it
    :param songs_list : List of urls
    :return: Saves the songs in mp3 format in the working directory
    """
    for song in songs_list:
        for s in song:
            video_url = s
            print(video_url)
            video_info = youtube_dl.YoutubeDL().extract_info(
                url = video_url,download=False)
            filename = dest+os.sep+f"{video_info['title']}.mp3"
            options={
                'format':'bestaudio/best',
                'keepvideo':False,
                'outtmpl':filename,
            }

            with youtube_dl.YoutubeDL(options) as ydl:
                ydl.download([video_info['webpage_url']])

            print("Download complete... {}".format(filename))

func_dict={
    'read_txt':read_txt,
    'read_csv':read_csv
}

if __name__=='__main__':
    path = sys.argv[1]
    dest = sys.argv[2]
    ext = path.rsplit('.')[1]
    songs_list = func_dict['read_'+ext](path)
    run(songs_list)
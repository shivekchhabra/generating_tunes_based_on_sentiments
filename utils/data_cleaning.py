import sys
import os
from pydub import AudioSegment
from mido import MidiFile


def convert(input_mp3, output_dest):
    """
    General function to convert mp3 to wav
    :param input_mp3: Input path + file (mp3)
    :param output_dest: Output path
    :return: Saves the tunes in wav format in the destination
    """
    sound = AudioSegment.from_mp3(input_mp3)
    filename = input_mp3.split('/')[-1].split('.')[0] + '.wav'
    sound.export(output_dest + os.sep + filename, format="wav")


def clipping_sound(sound, duration=30, cuts=-1):
    """
    General function to trim a tune down to 30 seconds.
    :param sound: Input sound
    :param duration: Duration of the clipping in seconds
    :param cuts: Number of cuts to be made
    :return: Trimmed audio clips
    """
    sound_clips = []
    first_cut_point = 0
    clip_size = duration * 1000
    last_cut_point = duration * 1000
    total_duration = sound.duration_seconds

    if cuts == -1:
        while total_duration >= duration:
            sound_clip = sound[first_cut_point:last_cut_point]
            first_cut_point = last_cut_point
            last_cut_point += clip_size
            total_duration -= clip_size / 1000
            sound_clips.append(sound_clip)
    else:
        while cuts > 0 and total_duration >= duration:
            sound_clip = sound[first_cut_point:last_cut_point]
            first_cut_point = last_cut_point
            last_cut_point += clip_size
            total_duration -= clip_size / 1000
            cuts -= 1
            sound_clips.append(sound_clip)
    return sound_clips


def mp3_to_wav(input_folder, output_dest):
    """
    General function to convert mp3 to wav.
    :param input_folder: Input folder where the MP3 songs are present
    :param output_dest: Output folder where wav songs will be saved
    :return: Converts and exports tunes to destination folder
    """
    for i in os.listdir(input_folder):
        if i.endswith('mp3'):
            convert(os.getcwd() + os.sep + input_folder + os.sep + i, output_dest)


def trim_audio(input_folder, output_dest, duration=30, cuts=10):
    """
    General function to trim audio based on a particular duration
    :param input_folder: Input tunes - mp3/wav
    :param output_dest: Output folder to save trimmed tunes
    :param duration: Duration to crop into
    :return: Trims and exports tunes to destination folder
    """

    """
    PS in my experience, I feel if you have mp3 files,
    run mp3_to_wav first and then run this for better results 
    """
    # Reading inputs from a folder.
    for i in os.listdir(input_folder):
        if i.endswith('mp3') or i.endswith("wav"):
            sound = AudioSegment.from_file(input_folder + os.sep + i)
            sound_clips = clipping_sound(sound, cuts=cuts, duration=duration)
            name = i.replace('.mp3', '.wav')
            # Saving the output to the output folder
            for clip in range(len(sound_clips)):
                sound_clips[clip].export(output_dest + os.sep + str(clip) + "_" + name.replace(" ", ""),
                                         format="wav")

def wav_to_midi(input_folder):
    """
    General function to convert wav to midi.
    :param input_folder: Input folder where the Wav songs are present
    :return: Converts and exports tunes to working directory
    """
    path = str(input_folder)

    os.chdir(path)
    audio_files = os.listdir()
    for file in audio_files:
        name, ext = os.path.splitext(file)
        if ext == ".wav":
            os.system("audio-to-midi %s.wav  -b 150 -t 60" % name)


def midi_to_wav(inputfolder):
    """
    General function to convert midi to wav.
    :param input_folder: Input folder where the Wav songs are present
    :return: Converts and exports tunes to directory
    """
    path = str(inputfolder)
    
    os.chdir(path)
    audio_files = os.listdir()
    for file in audio_files:
        name, ext = os.path.splitext(file)
        if ext == ".mid":
            mid = MidiFile(file)
            output = AudioSegment.silent(mid.length * 1000.0)
            output.export(name+".wav", format="wav")

if __name__ == '__main__':
    input_folder = input()
    output_dest = input()
    # mp3_to_wav(input_folder, output_dest)
    trim_audio(input_folder, output_dest)

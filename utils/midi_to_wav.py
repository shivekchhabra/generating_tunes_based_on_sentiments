import sys,os
from midi2audio import FluidSynth

def midi_to_wav(path):
    files=os.listdir(path)
    for f in files:
        if(f.split('.')[-1]=='mid' or f.split('.')[-1]=='midi'):
            print("converting "+f)
            FluidSynth().midi_to_audio(os.path.join(path,f),os.path.join(path,f+'-output.wav'))

if __name__ == '__main__':
    path = sys.argv[1]
    midi_to_wav(path)

# pip install h5py==2.10.0 pandas==1.1.2 librosa==0.6.0 numba==0.48 mido==1.2.9 mir_eval==0.5 matplotlib==3.0.3 torchlibrosa==0.0.4 sox==1.4.0
# pip install piano_transcription_inference

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

# Load audio
(audio, _) = load_audio("~/tempfolder/new_song.mp3", sr=sample_rate, mono=True)

# Transcriptor
transcriptor = PianoTranscription(device='cpu')    # 'cuda' | 'cpu'

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'trial.mid')


# if __name__ == '__main__':
#     path = sys.argv[1]
#     dest = sys.argv[2]
#     ext = path.rsplit('.')[1]
#     songs_list = func_dict['read_'+ext](path)
#     run(songs_list)
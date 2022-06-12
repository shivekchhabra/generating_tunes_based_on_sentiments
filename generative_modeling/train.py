import os
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint



# Parses all midi files in a given folder 

def get_notes(folder_name,notes):
    files=os.listdir(folder_name)
    for f in files:
        if(f.split('.')[-1]=='mid' or f.split('.')[-1]=='midi'):
            print("parasing "+f)
            midi = converter.parseFile(os.path.join(folder_name,f))
            a = instrument.partitionByInstrument(midi)
            if(len(a.parts)>1):
                print(f+" Contains more than one instruments recommended to replace this with better midi song, currently not used for training")
                continue
            parsed = a[0].recurse()
        for element in parsed:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('data/notes-'+folder_name, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


# Creates IO map which vectorizes notes with the corresponding emotion  

def create_io(inputs,outputs,emotion,note_map,notes):
    sample_len=100
    emotion_map={'happy':[0,0,1],'sad':[0,1,0],'thriller':[1,0,0]}
    for i in range(0,len(notes)-sample_len):
        inp_li=notes[i:i + sample_len]
        sample_list=[]
        for j in inp_li:
            li=[]
            li.append(note_map[j])
            for k in emotion_map[emotion]:
                li.append(k)
            sample_list.append(li)
        inputs.append(sample_list)
        out_note=notes[i + sample_len]

        #Either use the commented code or the default one, if experimenting with below commented code comment out last line
        # li=[]
        # li.append(note_map[out_note])
        # for j in emotion_map['emotion']:
        #     li.append(j)
        # outputs.append(li)

        outputs.append(note_map[out_note])
        

# Model architecture to train

def get_model(input,note_set_length):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(input.shape[1],input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(note_set_length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

# Function which trains the created model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

sample_len=100
notes=[]
get_notes('happy_songs',notes)
happy_offset = len(notes)
get_notes('sad_songs',notes)
sad_offset = len(notes)
get_notes('thriller_songs',notes)
notes_set=sorted(set(notes))
note_map={}
temp=0
for i in notes_set:
    note_map[i]=temp
    temp+=1
inputs=[]
outputs=[]
create_io(inputs,outputs,'happy',note_map,notes[0:happy_offset])
create_io(inputs,outputs,'sad',note_map,notes[happy_offset:])
create_io(inputs,outputs,'thriller',note_map,notes[sad_offset:])
# print("total notes are "+str(len(notes)))
print("number of inputs is "+ str(len(inputs)))
print("number of outputs is "+str(len(outputs)))
# print(outputs)
inputs=numpy.reshape(inputs,(len(inputs),sample_len, 4))
outputs = np_utils.to_categorical(outputs)
inputs=inputs/float(len(notes_set))
print(outputs.shape)
print(len(notes_set))
model = get_model(inputs,len(notes_set))
train(model,inputs,outputs)
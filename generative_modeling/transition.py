import os
import pickle
import numpy
import sys
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from generate import create_network,create_midi,create_io

def get_notes(f,notes):
    if(f.split('.')[-1]=='mid' or f.split('.')[-1]=='midi'):
        print("parasing "+f)
        midi = converter.parseFile(f)
        a = instrument.partitionByInstrument(midi)
        parsed = a[0].recurse()
    for element in parsed:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def get_stored_notes():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes-'+'sad_songs', 'rb') as filepath:
        total_notes = pickle.load(filepath)
    return total_notes



def generate_notes(model, initial, pitchnames, n_vocab,emotion,pred_initial):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    emotion_map={'happy':[0,0,1],'sad':[0,1,0],'thriller':[1,0,0]}

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = initial[-1]
    print("this is pattern shape, ",pattern[0])
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 4))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        li=[]
        li.append(index)
        for i in emotion_map[emotion]:
            li.append(i)
        pattern.append(li)
        pattern = pattern[1:len(pattern)]

    return prediction_output

if __name__=='__main__':
    sample_len=100
    path = sys.argv[1]
    emotion = sys.argv[2]
    notes=[]
    get_notes(path,notes)
    print('note shape ',len(notes))
    total_notes=get_stored_notes()
    print("total notes size ",len(total_notes))
    notes_set=sorted(set(total_notes))
    note_map={}
    temp=0
    for i in notes_set:
        note_map[i]=temp
        temp+=1
    inputs=[]
    create_io(inputs,emotion,note_map,notes)
    print("*************************************************************************************************************************")
    print("original input shape ",len(inputs))
    norm_inputs=numpy.reshape(inputs, (len(inputs), sample_len, 4))
    norm_inputs=norm_inputs/float(len(notes_set))
    model=create_network(norm_inputs,len(notes_set))
    # output=generate_notes(model,inputs,note_map,len(notes_set),'sad')
    output=generate_notes(model,inputs,note_map,len(notes_set),emotion,inputs)
    create_midi(output)



    
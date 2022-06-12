import os
import os
import pandas as pd
import sys
from joblib import dump, load
from midi2audio import FluidSynth
from mido import MidiFile
from pydub import AudioSegment
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from emotion_classification_model import EmotionClassifier, feature_names


def midi_to_wav(path):
    files=os.listdir(path)
    for f in files:
        if(f.split('.')[-1]=='mid' or f.split('.')[-1]=='midi'):
            FluidSynth().midi_to_audio(path + "/" + f, path + "/" + f+'-output.wav')


def interactive_prediction():
    path_1 = ""
    ec = EmotionClassifier()
    df = pd.DataFrame(columns=list(feature_names))
    feature_file_present = input("Do you already have the extracted features in a file ? [y | n] : ")
    if feature_file_present == "n":
        value = input("Do you want to convert \".mid\" files to \".wav\" ? [y | n] : ")

        if value == "y":
            path_1 = input("Specify the folder path to convert midi to wav : ")
            midi_to_wav(path_1)

        if path_1 == "":
            path_1 = input("Enter the folder path to extract features : ")
        print("Extracting features...")
        df, song_list = ec.add_features(df, path_1, "none", ".wav", [])
        feature_file = input("Specify the file name to store features, like predict_features_<number>.csv : ")
        df.to_pickle("artefacts" + "/" + feature_file)

        with open("artefacts" + "/" + "song_list_" + str(df.shape[0]) + str(len(df.columns)) + ".pkl", "wb") as f:
            pickle.dump(song_list, f)

    elif feature_file_present == "y":
        feature_file = input("Specify the feature file : ")

    song_list_file = input("Specify song list file : ")

    df = pd.DataFrame(columns=list(feature_names))
    df = pd.read_pickle("artefacts" + "/" + str(feature_file))
    df['indexcol'] = range(1, df.shape[0] + 1)

    # df = df[df["label"] != "happy"]

    le = LabelEncoder()
    df = df[df["label"] != "anger"]
    df["label_encoded"] = le.fit_transform(df["label"])
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    label = df["label_encoded"]
    df = df.drop(columns=["label", "label_encoded"])
    df = df.fillna(0)
    df["featurize"] = df.apply(ec.featurize, axis=1)

    model_name = input("Enter the model name to predict against")
    model = load("models" + "/" + model_name)
    y_pred_prob = model.predict_proba(list(df["featurize"]))
    y_pred = model.predict(list(df["featurize"]))
    print("Happy: 0, Sad: 1, Thriller: 2")
    print()
    with open("artefacts" + "/" + song_list_file, 'rb') as f:
        song_list = pickle.load(f)

    result_file = input("Enter file name to store result, like result.pkl : ")
    result_list = []
    for i in range(len(y_pred)):
        print(song_list[i], y_pred[i], y_pred_prob[i])
        result_list.append((song_list[i], y_pred[i], y_pred_prob[i]))

    with open("artefacts" + "/" + result_file, "wb") as result_file:
        pickle.dump(result_list, result_file)


def automated_prediction(path_1):
    ec = EmotionClassifier()
    midi_to_wav(path_1)
    df = pd.DataFrame(columns=list(feature_names))
    df, song_list = ec.add_features(df, path_1, "none", ".wav", [])
    df['indexcol'] = range(1, df.shape[0] + 1)
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    label = df["label_encoded"]
    df = df.drop(columns=["label", "label_encoded"])
    df = df.fillna(0)
    df["featurize"] = df.apply(ec.featurize, axis=1)
    model = load("models/final_midi_random_equity_model.joblib")
    y_pred_prob = model.predict_proba(list(df["featurize"]))
    y_pred = model.predict(list(df["featurize"]))
    result_list = []
    for i in range(len(y_pred)):
        result_list.append((song_list[i], y_pred[i], y_pred_prob[i]))
    return result_list


if __name__ == "__main__":
    print(interactive_prediction())

# Make sure to install
# scikit-learn==1.0.1 to load pickled files properly
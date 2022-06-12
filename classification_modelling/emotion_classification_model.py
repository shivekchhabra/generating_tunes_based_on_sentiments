import featuretools as ft
import librosa
import numpy as np
import os
import os
import pandas as pd
import sklearn as sk
import string
import sys
import sys
from itertools import chain
from joblib import dump, load
from midi2audio import FluidSynth
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

feature_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p','q', 'r',
                 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah',
                 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'at', 'au', 'av', 'aw', 'ax',
                 'ay', 'az', 'ba','bb', 'label']

df = pd.DataFrame(columns=list(feature_names))

def midi_to_wav(path):
    files=os.listdir(path)
    for f in files:
        if(f.split('.')[-1]=='mid' or f.split('.')[-1]=='midi'):
            FluidSynth().midi_to_audio(path + "/" + f, path + "/" + f+'-output.wav')


class EmotionClassifier:
    def compute_statistical_features_from_extracted_feature(self, ext_feature):
        avg_ext_feature = np.mean(ext_feature)
        std_ext_feature = np.std(ext_feature)
        var_ext_feature = np.var(ext_feature)
        return [avg_ext_feature, std_ext_feature, var_ext_feature]


    def add_features(self, df, path, label, format, song_list):
        print(path, "______________________________________")
        for song in os.listdir(path):
            if song.endswith(format):
                print(path, song)
                music_file, sampling_rate = librosa.load(os.path.join(path, song))
                short_time_fourier = np.abs(librosa.stft(music_file))
                tempo, beats = librosa.beat.beat_track(music_file, sr=sampling_rate)
                chromagram_stft = librosa.feature.chroma_stft(music_file, sr=sampling_rate)
                chromagram_cq = librosa.feature.chroma_cqt(music_file, sr=sampling_rate)
                chroma_cens = librosa.feature.chroma_cens(music_file, sr=sampling_rate)
                melspectrogram = librosa.feature.melspectrogram(music_file, sr=sampling_rate)
                cent = librosa.feature.spectral_centroid(music_file, sr=sampling_rate)
                spec_bw = librosa.feature.spectral_bandwidth(music_file, sr=sampling_rate)
                contrast = librosa.feature.spectral_contrast(S=short_time_fourier, sr=sampling_rate)
                rolloff = librosa.feature.spectral_rolloff(music_file, sr=sampling_rate)
                poly_features = librosa.feature.poly_features(S=short_time_fourier, sr=sampling_rate)
                tonnetz = librosa.feature.tonnetz(music_file, sr=sampling_rate)
                zcr = librosa.feature.zero_crossing_rate(music_file)
                harmonic = librosa.effects.harmonic(music_file)
                percussive = librosa.effects.percussive(music_file)
                mfcc = librosa.feature.mfcc(music_file, sr=sampling_rate)
                mfcc_delta = librosa.feature.delta(mfcc)
                onset_frames = librosa.onset.onset_detect(music_file, sr=sampling_rate)
                frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sampling_rate)

                features = []

                beats_sum = sum(beats)
                avg_beats = np.average(beats)

                features.extend([beats_sum, avg_beats])

                extracted_features = [chromagram_stft, chromagram_cq, chroma_cens, melspectrogram,
                                      cent, spec_bw, contrast, rolloff, poly_features, tonnetz,
                                      zcr, harmonic, percussive, mfcc, mfcc_delta, onset_frames, frames_to_time]

                for extracted_feature in extracted_features:
                    features.extend(self.compute_statistical_features_from_extracted_feature(extracted_feature))

                features.extend([label])
                row = dict()
                for index in range(len(features)):
                    row[feature_names[index]] = features[index]
                song_list.append(song)
                df = df.append(row, ignore_index=True)
        return df, song_list

    def featurize(self, entry):
        return tuple(entry)


    def featureImpl(self, df):
        es = ft.EntitySet(id="Music Classification")
        es.add_dataframe(dataframe_name='music',
                         dataframe=df,
                         index="indexcol")

        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='music')
        return feature_matrix

    def accuracy_metrics(self, y_test, y_pred):
        happy_classified = 0
        sad_classified = 0
        thriller_classified = 0

        happy_total_count = 0
        sad_total_count = 0
        thriller_total_count = 0

        y_test_list = list(y_test)
        for i in range(len(y_test_list)):
            if y_test_list[i] == 0:
                happy_total_count += 1
            if y_test_list[i] == 1:
                sad_total_count += 1
            if y_test_list[i] == 2:
                thriller_total_count += 1

            if y_test_list[i] == 0 and y_pred[i] == 0:
                happy_classified += 1
            if y_test_list[i] == 1 and y_pred[i] == 1:
                sad_classified += 1
            if y_test_list[i] == 2 and y_pred[i] == 2:
                thriller_classified += 1

        happy_accuracy = happy_classified / happy_total_count
        sad_accuracy = sad_classified / sad_total_count
        thriller_accuracy = thriller_classified / thriller_total_count

        return happy_accuracy, sad_accuracy, thriller_accuracy

if __name__ == "__main__":
    ec = EmotionClassifier()

    feature_file_present = input("Do you already have the extracted features in a file ? [y | n] : ")
    if feature_file_present == "n":

        print("Specify configurations for the 3 genres (Happy/Sad/Thriller) : ")
        path_1 = input("Folder path with happy songs: ")
        label_1 = "happy"
        path_2 = input("Folder path with sad songs: ")
        label_2 = "sad"
        path_3 = input("Folder path with thriller songs: ")
        label_3 = "thriller"

        value = input("Do you want to convert \".mid\" files to \".wav\" ? [y | n] : ")

        if value == "y":
            # The songs have be to segregated in separate folders, each for one genre

            midi_to_wav(path_1)
            midi_to_wav(path_2)
            midi_to_wav(path_3)

        feature_flag = input("Do you want to proceed with feature extraction for the files ? [y | n] : ")
        if feature_flag == "y":
            print("Extracting features...")

            # Feature extraction per folder(genre) along with the label for that folder and
            # file format to be considered. Default file format for now : ".wav"

            df, song_list = ec.add_features(df, path_1, label_1, ".wav", [])
            df, song_list = ec.add_features(df, path_2, label_2, ".wav", song_list)
            df, song_list = ec.add_features(df, path_3, label_3, ".wav", song_list)

            # Once features are extracted, dump them in a pickle file to re-use later

            features_file = input("Enter file name to dump the features, like : \"features_extracted_<number>.csv\" : ")
            df.to_pickle("artefacts" + "/" + features_file)
            with open("artefacts" + "/" + "song_list_" + str(df.shape[0]) + str(len(df.columns))  + ".pkl", "wb") as f:
                pickle.dump(song_list, f)
    elif feature_file_present == "y":
        feature_file = input("Enter the name of the feature file : ")

    df = pd.read_pickle("artefacts" + "/" + feature_file)
    df = df[df["label"] != "anger"]

    df['indexcol'] = range(1, df.shape[0] + 1)
    # df_ft = ec.featureImpl(df)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print(le_name_mapping)

    label = df["label_encoded"]
    df = df.drop(columns=["label", "label_encoded"])

    df["featurize"] = df.apply(ec.featurize, axis=1)

    result_model = ""
    threshold_accuracy = 0.0

    print("Building your model... please wait....")

    for i in range(25):
        x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(df, label, test_size=0.3)

        #     print(x_train.shape[0], x_train.shape[1])
        #     print(x_test.shape[0], x_test.shape[1])
        #     print(y_train.shape[0])
        #     print(y_test.shape[0])

        train_list = list(x_train["featurize"])
        model = RandomForestClassifier().fit(train_list, y_train)
        y_pred = model.predict(list(x_test["featurize"]))
        score = accuracy_score(y_test, y_pred)
        if score > threshold_accuracy:
            result_model = model
            threshold_accuracy = score
            happy_accuracy, sad_accuracy, thriller_accuracy = ec.accuracy_metrics(y_test, y_pred)

    print("Overall validation accuracy", threshold_accuracy)
    print()
    print("Happy validation accuracy", happy_accuracy)
    print("Sad validation accuracy", sad_accuracy)
    print("Thriller validation accuracy", thriller_accuracy)

    x_pred = result_model.predict(list(x_train["featurize"]))
    happy_accuracy, sad_accuracy, thriller_accuracy = ec.accuracy_metrics(list(y_train), x_pred)
    print()
    print("Happy train accuracy", happy_accuracy)
    print("Sad train accuracy", sad_accuracy)
    print("Thriller train accuracy", thriller_accuracy)

    save_model = input("Do you want to save this model ? [y | n] : ")
    if save_model == "y":
        model_name = input("Enter the model file name to store the model : ")
        dump(result_model,  "models/" + model_name + ".joblib")

# Make sure to install
# scikit-learn==1.0.1 to load pickled files properly

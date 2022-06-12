# Description:
This is a webapp for CMPT-733 Big Data programming II.

In this a user can generate totally new tunes at each refresh by just choosing an emotion.

The tune is generated using a LSTM model and can then be classified using the classification model for by passing the subjectivity.

This classification model can then be used to refine the results of the generative model

# Usage:

## Running the app:

To run the app, you need to open 2 terminal windows.

1. In one terminal window, run the generator pipeline. This will keep a check on the "generated_songs" folder. If the songs are less than 10, it will start generating.:
<pre><code>python generate_pipeline.py</code></pre>

3. In the 2nd terminal window, run the app:
<pre><code>python app.py -p 5050</code></pre>
here the port is optional (default - 5000)

Now you can go on the browser and on localhost:5050, you will see your app running.

### Acknowledgements:

Librosa feature transformation of song input has been inspired by the works of Danyal Imran (Data Analyst, Berlin)

### References:
1) https://en.wikipedia.org/wiki/[Music_and_emotion](Music_and_emotion)
2) [https://ismir.net/](International Society for Music Information Retrieval Conference)

## For EDA:

### How to run:

To run mp3_to_wav, and trim_tunes, you need to install pydub from requirements.txt and setting up ffmpeg

<pre><code>pip install pydub</code></pre>

and

#### For Mac:

<pre><code>brew install ffmpeg</code></pre>

#### For Ubuntu:

<pre><code>apt-get install ffmpeg</code></pre>

### Piano mp3 to midi
please make sure to install the below dependancies before running the piano mp3 to midi convertor
<pre><code>pip install h5py==2.10.0 pandas==1.1.2 librosa==0.6.0 numba==0.48 mido==1.2.9 mir_eval==0.5 matplotlib==3.0.3 torchlibrosa==0.0.4 sox==1.4.0</code></pre>
<pre><code>pip install piano_transcription_inference</code><pre>

### Midi to Wav
Install the following dependancies
<pre><code>pip install fluidsynth</code></pre>
<pre><code>pip install midi2audio</code></pre>
The next installation will depend on your OS, ive provided instructions for mac
<pre><code>sudo port install fluidsynth</pre></code>
<pre><code>mkdir -p ~/.fluidsynth</pre></code>
After creating the directory copy the .sf2 file present in the dependancy folder to ~/.fluidsynth



import os
import argparse
import shutil
from flask import Flask, render_template, request, flash

app = Flask(__name__)


def argument_parser():
    """
    Setting up an argument parser for dynamic port changing
    :return: args
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default=5000,
                    help="port to run the app on")
    args = vars(ap.parse_args())
    return args


@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Homepage of the webapp
    :return: Renders the page
    """
    return render_template('index.html', data=[{'emotion': 'Please Select an Emotion'}, {'emotion': 'Happy'},
                                               {'emotion': 'Sad'}, {'emotion': 'Thriller'}])


@app.route("/generate", methods=['GET', 'POST'])
def generate():
    """
    The url redirect to generate API.
    Users can select emotion and generate tunes
    :return: Renders the generate page
    """
    input_data = list(request.form.values())
    emotion = input_data[0].lower()
    path = ''
    try:
        if emotion in os.listdir("static/generated_songs"):
            for i in os.listdir(f'static{os.sep}generated_songs' + os.sep + emotion):
                if i.endswith("mid") or i.endswith("midi"):
                    print('generated_songs' + os.sep + emotion + os.sep + i)
                    shutil.copyfile(f'static{os.sep}generated_songs' + os.sep + emotion + os.sep + i, f"static{os.sep}staging" + os.sep + i)
                    path = "staging" + os.sep + i
                    os.remove(f'static{os.sep}generated_songs' + os.sep + emotion + os.sep + i)
                    break
        else:
            return render_template('index.html',
                                   data=[{'emotion': 'Please Select an Emotion'}, {'emotion': 'Happy'},
                                         {'emotion': 'Sad'}, {'emotion': 'Thriller'}])
    except Exception as e:

        print("Exception: {}".format(str(e)))

    return render_template('generate.html', path=path,
                           data=[{'emotion': 'Please Select an Emotion'}, {'emotion': 'Happy'},
                                 {'emotion': 'Sad'}, {'emotion': 'Thriller'}])


if __name__ == '__main__':
    args = argument_parser()
    app.run(debug=True, port=int(args['port']))

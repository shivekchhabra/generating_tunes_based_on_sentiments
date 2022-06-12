import time

from generative_modeling.generate import *
import os
import uuid

sep = os.sep


def pipeline_runner(notes_folder, name, category, sample_len=100):
    """
    Pipeline code to run the predict functionality
    :param notes_folder: The folder where data notes are located
    :param name: Name for the output file
    :param category: Category of the tune to generate
    :param sample_len: Length of notes to be taken as sample; default 100
    :return:
    """
    if 'generated_songs' not in os.listdir():
        os.mkdir("generated_songs")
    if category not in os.listdir("generated_songs"):
        os.mkdir("generated_songs" + sep + category)

    notes, total_notes = get_notes(notes_folder)
    notes_set = sorted(set(total_notes))
    note_map = {}
    temp = 0
    for i in notes_set:
        note_map[i] = temp
        temp += 1
    inputs = []
    create_io(inputs, category, note_map, notes)
    norm_inputs = numpy.reshape(inputs, (len(inputs), sample_len, 4))
    norm_inputs = norm_inputs / float(len(notes_set))
    model = create_network(norm_inputs, len(notes_set))

    output = generate_notes(model, inputs, note_map, len(notes_set), category)
    stream = create_midi(output)
    stream.write('midi', fp=f'static{sep}generated_songs{sep}{category}{sep}{name}.mid')


def checker():
    """
    Continuously monitors the app every x seconds and generates as required.
    :return:
    """
    while (1):
        time.sleep(10)
        if 'generated_songs' not in os.listdir("static"):
            os.mkdir(f"static{sep}generated_songs")
            for emo in ['happy', 'sad', 'thriller']:
                os.mkdir(f"static{sep}generated_songs" + sep + emo)
                for i in range(10):
                    pipeline_runner(notes_folder=f"{emo}_songs", name=uuid.uuid4(), category=emo)
                    print(f"Song generated for {emo}")

        else:
            for folders in os.listdir(f"static{sep}generated_songs"):
                if folders in ['happy', 'sad', 'thriller']:
                    data = list(os.listdir(f"static{sep}generated_songs" + sep + folders))
                    if '.DS_Store' in data:
                        data.remove('.DS_Store')
                    if len(data) < 10:
                        for i in range(10 - len(data)):
                            print(f"Generating a {folders} tune")
                            pipeline_runner(notes_folder=f"{folders}_songs", name=uuid.uuid4(), category=folders)
                            print(f"Song generated for {folders}")
                    else:
                        print("Not generating... passing")
                        pass


if __name__ == '__main__':
    checker()

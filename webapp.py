# Mark Evers
# Created: 4/9/2018
# webapp.py
# Web application


import os
import sys
sys.path.append(os.getcwd())
sys.path.append("src")


from flask import Flask, render_template, send_from_directory, request, flash
from werkzeug.utils import secure_filename
from src.file_handlers.dataset import VectorGetterNHot
from src.model_final import load_from_disk
import numpy as np
from src.file_handlers.midi_archive import MidiArchive
from src.midi_handlers.midi_file import MidiFileNHot


app = Flask(__name__)
app.secret_key = os.urandom(24)
composers = VectorGetterNHot("midi/classical").composers
upload_folder = "temp_midi_uploads"

model = load_from_disk("models/final")
graph = model.get_default_graph


ALLOWED_EXTENSIONS = {"mid", "midi", "MID", "MIDI"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_one_file(filename):

    global graph

    # model = load_from_disk("models/final")
    # model._make_predict_function()
    note_dist = MidiArchive.parse_midi_meta(filename)[14:]

    mid = MidiFileNHot(filename, note_dist)
    X = np.array(mid.to_X(), dtype=np.byte)

    with graph.as_default():
        y_pred = model.predict(X)
    sum_probs = y_pred.sum(axis=0)
    normed_probs = sum_probs / sum_probs.sum()

    prediction = composers[np.argmax(normed_probs)]

    return prediction, normed_probs



@app.route("/index.html", methods=['GET'])
@app.route("/", methods=['GET'])
def index():
    return render_template("shell.html", content="index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/composers.html')
def show_composers():
    return render_template("shell.html", content="composers.html")

@app.route('/contact.html')
def contact():
    return render_template("shell.html", content="contact.html")

@app.route('/midi.html', methods=['GET', 'POST'])
def midi():

    print(request.method)

    if request.method == "GET":
        return render_template("shell.html", content="upload_form.html")

    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template("shell.html", content="upload_form.html")

        file = request.files['file']

        # submitted with empty filename
        if file.filename == '':
            flash('No selected file')
            return render_template("shell.html", content="upload_form.html")

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            temp_midifile_path = os.path.join(upload_folder, filename)
            file.save(temp_midifile_path)

            # try:
            #     prediction, probs = predict_one_file(temp_midifile_path)
            # except:
            #     return render_template("shell.html", content="corrupt.html")
            # finally:
            #     os.remove(temp_midifile_path)

            prediction, probs = predict_one_file(temp_midifile_path)
            os.remove(temp_midifile_path)

            return render_template("shell.html", content="midi.html", filename=filename, prediction=prediction, probs=probs, composers=composers, probs_i=np.argsort(probs)[::-1])


        return render_template("shell.html", content="corrupt.html")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

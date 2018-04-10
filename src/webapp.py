# Mark Evers
# Created: 4/9/2018
# webapp.py
# Web application


from flask import Flask, render_template, send_from_directory, request, flash, redirect
import os
from werkzeug.utils import secure_filename
from file_handlers.dataset import VectorGetterNHot
import pickle
from model_final import load_from_disk, predict_one_file


app = Flask(__name__)
model = load_from_disk("models/final")
dataset = VectorGetterNHot("midi/classical")



ALLOWED_EXTENSIONS = {".mid", ".midi", ".MID", ".MIDI"}



@app.route("/index.html", methods=['GET'])
@app.route("/", methods=['GET'])
def index():
    return render_template("shell.html", content="index.html")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/midi.html", methods=['GET', 'POST'])
def midi():

    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

    return render_template("shell.html", content="midi.html")


@app.route('/midi.html', methods=['GET', 'POST'])
def upload_file():

    if request.method == "GET":
        return render_template("shell.html", content="upload_form.html")

    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)


        file = request.files['file']

        # submitted with empty filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            midifile_path = os.path.join("temp_midi_uploads", filename)
            file.save(midifile_path)


            prediction, probs = predict_one_file(dataset, model, midifile_path)
            prediction = dataset.composers[prediction]

            return render_template("shell.html", content="midi.html", filename=filename, prediction=prediction, probs=probs)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


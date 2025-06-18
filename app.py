from flask import Flask, request, jsonify
import whisper
import tempfile

app = Flask(__name__)
model = whisper.load_model("base")  # Use "tiny" if performance is slow

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        audio.save(temp_file.name)
        result = model.transcribe(temp_file.name)

    return jsonify({"text": result["text"]})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT") or 5000)
    app.run(host='0.0.0.0', port=port)

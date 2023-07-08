from flask import Flask, render_template, request
import os
import uuid
import moviepy.editor as mp
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from langdetect import detect
from googletrans import Translator
import shutil

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the uploaded video or video link and target language
    video = request.files.get('video')
    link = request.form.get('link')
    target_language = request.form.get('target-language')

    if video:
        # Save the uploaded video to a temporary file
        filename = str(uuid.uuid4()) + '.mp4'
        video.save(filename)
    elif link:
        # Download the video from the link to a temporary file
        filename = "C:\\Users\\hp\\Desktop\\python_project\\test_video_french.mp4"
        os.system(f'youtube-dl -o "{filename}" {link}')
    else:
        # Return an error message if no video is provided
        return 'Error: Please upload a video or paste a video link.'

    # Extract the audio content from the video
    video_clip = mp.VideoFileClip(filename)
    audio = video_clip.audio
    audio.write_audiofile("temp.wav")
    video_clip.close()
    
    # Convert the audio content to text
    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data)

    # Detect the language of the text
    detected_language = detect(text)

    # Translate the text to the target language if different from detected language
    if detected_language != target_language:
        translator = Translator()
        translated_text = translator.translate(text, dest=target_language).text
    else:
        translated_text = text

    # Load the BERT summarization model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    # Generate a summary of the translated text
    inputs = tokenizer.encode(translated_text, return_tensors="pt")
    max_length = inputs.size(1)  # Get the actual length of the tensor
    outputs = model.generate(inputs, max_length=max_length + 1, min_length=10, num_beams=5)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Delete the temporary files
    os.remove("temp.wav")

    # Move the video file to a different location and then delete it
    temp_location = "C:\\Users\\hp\\Desktop\\python_project\\temp_video.mp4"
    shutil.move(filename, temp_location)
    os.remove(temp_location)

    # Return the summary as the video description
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)

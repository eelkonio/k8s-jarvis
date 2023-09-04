#!/usr/bin/env python3

import re
import os
from datetime import datetime
import time
import struct

from google.cloud import texttospeech
import pvleopard as pvleopard
import pvporcupine
import pyaudio
import wave
import openai
import pyttsx3
import numpy as np
from openwakeword.model import Model

import PySimpleGUI as sg


voices=["sam","pyttsx3"]
voice_idx=1  # start with sam!

first_run=True

#   help="The inference framework to use (either 'onnx' or 'tflite'",
inference_framework='tflite'
# Load pre-trained openwakeword models
#owwModel = Model(wakeword_models=[model_path], inference_framework=inference_framework)
owwModel = Model(inference_framework=inference_framework)

n_models = len(owwModel.models.keys())


USE_GOOGLE_VOICE=False

#my_keywords=['jarvis', 'terminator']

PORCUPINE_ACCESS_KEY=os.environ.get("PORCUPINE_ACCESS_KEY", "UNKNOWN")
openai.api_key = os.environ.get("openai_api_key", "UNKNOWN")
if PORCUPINE_ACCESS_KEY=="UNKNOWN":
    print ("Please set your environment correctly, dude.")
    exit(0)

sample_rate=16000
frame_length=512

#porcupine = pvporcupine.create(
#    access_key=PORCUPINE_ACCESS_KEY,
#    keywords=my_keywords
#)

leopard = pvleopard.create(access_key=PORCUPINE_ACCESS_KEY)

# Initialize the voice library
engine = pyttsx3.init()



# Set up a nice window for our output
layout = [
    [sg.Multiline('', key='my-chat', size=(100, 30), autoscroll=True)],
    [sg.Button('Exit')]
]
window = sg.Window( 'scrollable-chat', layout )
event, values = window.read( timeout=10  )  # Wait for a few moments for an event





def close_program():
    global window
    #global porcupine
    #global stream
    #global audio


    # Clean up resources
    window.close()
    #porcupine.delete()
    #stream.stop_stream()
    #stream.close()
    #audio.terminate()
    exit(0)



def gui_print(text):
    global window
    window['my-chat'].print(text, text_color='black', background_color='white')
    event, values = window.read(timeout=10)  # Wait for up to 1 second for an event
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        close_program()



# Run capture loop continuosly, checking for wakewords
def wait_for_wakeword():
    global first_run
    global window
    
    #    help="How much audio (in samples) to predict on at once",
    chunk_size=1280

    #    help="The path of a specific model to load",
    model_path=""

    # Get microphone stream
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = chunk_size
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Skip clearing the audio buffer only in the first run
    if not first_run:
        # gui_print ("Clearing audio buffer")
        current_epoch=time.time()
        while int(time.time())<current_epoch+2:
            # Get audio
            audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
            # Feed to openWakeWord model
            prediction = owwModel.predict(audio)
    else:
        first_run=False

    # Generate output string header
    gui_print(f"Listening to your smegsy voice...")

    jarvis_called=False
    while not jarvis_called:
        now = datetime.now()
        
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        event, values = window.read(timeout=10)  # Wait for up to 1 second for an event
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            close_program()


        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])

            if mdl=="hey_jarvis" and scores[-1] >= 0.5:
                gui_print (f"keyword {mdl} is noticed in the latest scores.")
                jarvis_called=True
            elif scores[-1]>=0.5:
                gui_print(f"Noticed keyword {mdl}, but ignoring it.")

    gui_print ("Jarvis is called.")



def record_audio(filename, duration):
    frames = []


    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        rate=sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=frame_length * 2,
    )

    for _ in range(0, int(sample_rate / frame_length * duration)):
        audio_data = stream.read(frame_length, exception_on_overflow=False)
        audio_frame = struct.unpack_from("h" * frame_length, audio_data)
        frames.append(audio_data)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))



def synthesize_text(text, language_code="en-US", voice_name="en-US-Wavenet-D"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice_params, audio_config=audio_config
    )

    return response.audio_content


def play_audio(audio_content):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust the speech rate (words per minute)

    # Set the pyttsx3 engine to use the Google Text-to-Speech audio content
    def onStart(name):
        engine.audio.append(audio_content)

    engine.connect('started-utterance', onStart)
    engine.say(' ')
    engine.runAndWait()




def play_wav(fn):
    chunk=1024
    if not os.path.exists(fn):
        gui_print (f"ERROR: play_wav: no file found name {fn}")
        return
    wf = wave.open(fn, 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open pvstream based on the wave object which has been input.
    pvstream = p.open(format =
                      p.get_format_from_width(wf.getsampwidth()),
                      channels = wf.getnchannels(),
                      rate = wf.getframerate(),
                      output = True)
    
    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play pvstream (looping from beginning of file to the end)
    while data:
        # writing to the pvstream is what *actually* plays the sound.
        pvstream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    wf.close()
    pvstream.close()    
    p.terminate()



def speak(answer):
    if USE_GOOGLE_VOICE:
        gui_print(answer)
        audio_content = synthesize_text(answer)
        play_audio(audio_content)
    else:
        if voices[voice_idx]=="pyttsx3":
            # This pretty line of the code reads openAI response
            gui_print(answer)
            pyttsx3.speak(answer)
        elif voices[voice_idx]=="sam":
            tmpfile="/tmp/speech.wav"
            for sentence in answer.split("."):
                sam_sentence=re.sub("[^a-zA-Z0-9 ]","",sentence)
                if sam_sentence!="":
                    sam_command=f"sam -wav {tmpfile} {sam_sentence}."
                    gui_print (sentence)
                    os.system(sam_command)
                    play_wav(tmpfile)
                    os.remove(tmpfile)




# Main program


# Saying some fun welcome message with instructions for the user
welcome="I'm Jarvis. Ask me anything!"
gui_print (welcome)
engine.say(welcome)
engine.runAndWait()

stop_command=False
while not stop_command:
    #gui_print("Listening for keywords...")
    wait_for_wakeword()

    # Record speech for a fixed duration
    duration_seconds = 5
    audio_file = "recorded_audio.wav"
    #gui_print(f"Recording audio for {duration_seconds} seconds into {audio_file}")
    record_audio(audio_file, duration_seconds)

    # Transcribe the recorded speech using Leopard
    #gui_print("Transcribing speech...")
    transcript, words = leopard.process_file(os.path.abspath(audio_file))
    gui_print(f"You: '{transcript}'")

    if transcript=="Switch voice":
        gui_print ("*** SWITCHING VOICE ***")
        voice_idx+=1
        if voice_idx>len(voices):
            voice_idx=0
        speak(f"I have switched to voice {voices[voice_idx]}.")
        continue
    elif transcript=="Exit program":
        gui_print ("*** EXITING PROGRAM ***")
        speak(f"I will exit now. Bye, bye.")
        close_program()
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant",
                   "content": ("If applicable, use your Kubernetes and Cloud computing knowledge to formulate a short reply to this question: "+transcript)}],
        temperature=0.6,
    )

    answer=response.choices[0].message.content
    speak(answer)

    # Remove the audio file if you don't need it
    os.remove(audio_file)


close_program()




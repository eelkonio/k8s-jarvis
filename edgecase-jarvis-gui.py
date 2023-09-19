#!/usr/bin/env python3

import re
import os
from datetime import datetime
import time
import struct
import random

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


#----- TTS
# import all the modules that we will need to use
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

from playsound import playsound
tts_path = "/home/eelko/projects/k8s-jarvis/lib/python3.10/site-packages/TTS/.models.json"
model_manager = ModelManager(tts_path)
model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

audio_file = "recorded_audio.wav"
max_wait=3

tts_syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=voc_path,
    vocoder_config=voc_config_path
)

#------





voices=["tts","pyttsx3",'sam']
voice_idx=1  # start with tts

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



def gui_print(text,color):
    global window
    window['my-chat'].print(text, text_color=f'{color}', background_color='white')
    event, values = window.read(timeout=10)  # Wait for up to 1 second for an event
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        close_program()

last_jokes=""
def tell_one_joke():
    global last_jokes
    # start telling a joke
    rnd=random.randint(0,10)
    if rnd<2:
      joke=chatGPT("Tell me a new cloud-computing joke. Make it very uncommon, intelligent and funny. Do not repeat the following jokes: {last_jokes}", random.random())
    elif rnd<4:
      joke=chatGPT("Tell me a new, original programming joke. Make it intelligent and funny. Do not repeat the following jokes: {last_jokes}", random.random())
    elif rnd<6:
      joke=chatGPT("Tell me a new, original science joke. Make it intelligent and funny. Do not repeat the following jokes: {last_jokes}", random.random())
    else:
      joke=chatGPT(f"Combine original, new mom-jokes with kubernetes or cloud computing. Start the joke with 'Yo cluster is so big...'  Do not repeat the following jokes: {last_jokes}",
              random.random())
    joke=" ".join(joke.splitlines())
    last_jokes=f"{last_jokes}\n{joke}"
    if len(last_jokes)>20000:
        last_jokes_lines=last_jokes.splitlines()
        last_jokes="\n".join(last_jokes_lines[len(last_jokes_lines)//2])
    #print(f"LAST-JOKES: {last_jokes}\n\n")

def tell_some_jokes():
    # start telling a few jokes
    speak("This is boring. Let me tell you some jokes.")
    gui_print("","black")
    tell_one_joke()
    gui_print("","black")
    tell_one_joke()
    gui_print("","black")
    tell_one_joke()
    gui_print("","black")
    speak("I hope you liked the jokes. I am listening to you again.")



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
        current_epoch=time.time()
        while int(time.time())<current_epoch+2:
            # Get audio
            audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
            # Feed to openWakeWord model
            prediction = owwModel.predict(audio)
    else:
        first_run=False

    # Generate output string header
    gui_print(f"Listening to your smegsy voice...", 'black')

    jarvis_called=False
    start_time = datetime.now()
    while not jarvis_called:
        now = datetime.now()
        
        elapsed_delta=(now-start_time)
        elapsed=elapsed_delta.total_seconds() % max_wait

        if elapsed>(max_wait-0.5):
            tell_some_jokes()
            start_time=datetime.now()

        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        event, values = window.read(timeout=10)  # Wait for up to 1 second for an event
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            close_program()


        for mdl in owwModel.prediction_buffer.keys():
            #print(f"checking keyword {mdl}")
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])

            if mdl=="hey_jarvis" and scores[-1] >= 0.5:
                gui_print (f"keyword {mdl} is noticed in the latest scores.", 'black')
                jarvis_called=True
            elif scores[-1]>=0.5:
                gui_print(f"Noticed keyword {mdl}, but ignoring it.", 'black')

    gui_print ("Jarvis is called.", 'black')



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


def saveline(line):
    with open("savedlines.txt", mode='a') as f:
        f.write(f"{datetime.now()} -- {line}\n")



def play_wav(fn):
    chunk=1024
    if not os.path.exists(fn):
        gui_print (f"ERROR: play_wav: no file found name {fn}", 'red')
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
    saveline(answer)
    if USE_GOOGLE_VOICE:
        gui_print(answer,'blue')
        audio_content = synthesize_text(answer)
        play_audio(audio_content)
    else:
        if voices[voice_idx]=="pyttsx3":
            # This pretty line of the code reads openAI response
            gui_print(answer,'blue')
            pyttsx3.speak(answer)
        elif voices[voice_idx]=="sam":
            tmpfile="/tmp/speech.wav"
            for sentence in answer.split("."):
                sam_sentence=re.sub("[^a-zA-Z0-9 ]","",sentence)
                if sam_sentence!="":
                    sam_command=f"/usr/local/bin/sam -wav {tmpfile} {sam_sentence}."
                    gui_print (sentence,'blue')
                    os.system(sam_command)
                    play_wav(tmpfile)
                    os.remove(tmpfile)
        elif voices[voice_idx]=='tts':
            gui_print(answer,'blue')
            outputs = tts_syn.tts(answer)
            tts_syn.save_wav(outputs, "/tmp/audio-1.wav")
            playsound('/tmp/audio-1.wav')


def chatGPT(transcript, temperature):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "assistant",
                   "content": ("If applicable, use your Kubernetes and Cloud computing knowledge to formulate a short reply to this question: "+transcript)}],
        temperature=temperature,
    )

    answer=response.choices[0].message.content
    speak(answer)

    # Remove the audio file if you don't need it
    if os.path.isfile(audio_file):
      os.remove(audio_file)

    return answer

# Main program


# Saying some fun welcome message with instructions for the user
welcome="I'm Jarvis. Ask me anything!"
gui_print (welcome, 'blue')
engine.say(welcome)
engine.runAndWait()


stop_command=False
while not stop_command:
    #gui_print("Listening for keywords...", 'black')
    wait_for_wakeword()

    # Record speech for a fixed duration
    duration_seconds = 5
    #gui_print(f"Recording audio for {duration_seconds} seconds into {audio_file}", 'black')
    record_audio(audio_file, duration_seconds)

    # Transcribe the recorded speech using Leopard
    #gui_print("Transcribing speech...", 'black')
    transcript, words = leopard.process_file(os.path.abspath(audio_file))
    gui_print(f"You: '{transcript}'", 'black')
    saveline(transcript)

    transcript=transcript.lower()
    if 'voice' in transcript and 'switch' in transcript:
        gui_print ("*** SWITCHING VOICE ***", 'black')
        voice_idx+=1
        if voice_idx>=len(voices):
            voice_idx=0
        speak(f"I have switched to voice {voices[voice_idx]}.")
        continue
    elif 'joke' in transcript:
        tell_one_joke()
        continue
    elif 'exit' in transcript:
        gui_print ("*** EXITING PROGRAM ***", 'black')
        speak(f"I will exit now. Bye, bye.")
        close_program()
    
    chatGPT(transcript, 0.5)

close_program()




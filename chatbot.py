# Speech-to-text
import speech_recognition as sr
# Text-to-speech
from gtts import gTTS
# For the language model
import os
import time
import transformers
# For data
from datetime import datetime
import numpy as np
import os

# Setting up the bot
class ChatBot():
    def __init__(self, name):
        print("--------- Starting up", name, "----------")
        self.name = name
        
    # Speech to Text conversion
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(mic, duration=0.5) 
            audio = recognizer.listen(mic, timeout=2, phrase_time_limit=5)
        try:
            self.text = recognizer.recognize_google(audio)
            print("Me --> ", self.text)
        except sr.WaitTimeoutError:
            print("Me --> ERROR: Listening timed out after wating for the user to start speaking.")
            self.text = ""
        except sr.UnknownValueError:
            print("Me --> ERROR: Could not understand audio")
            self.text = ""
        except sr.RequestError:
            print("Me --> ERROR: Could not request results from Google Speech Recognition Service")
            self.text = ""
    
    # To activate the bot only when its name is spoken
    def wake_up(self, text):
        return True if self.name in text.lower() else False
    
    # Conversion from text to speech for bot responses
    @staticmethod
    def text_to_speech(text):
        print("AI --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")

        # Calculate audio file size and estimate its duration
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200

        # Play the MP3 file
        os.system("afplay res.mp3") ## need to add support for windows and linux

        # Wait for the duration of the audio to finish
        time.sleep(duration)

        # Remove the MP3 file after playing
        os.remove("res.mp3")

    # Function to respond with the current time in AM/PM format
    @staticmethod
    def action_time():
        current_time =  datetime.now().strftime('%I:%M %p')
        return f"The time right now is {current_time}."


# Starting the bot
if __name__ == "__main__":
    ai = ChatBot(name="neha")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex = True
    while ex:
        ai.speech_to_text()
        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Neha the AI, what can I do for you?"
        ## performing any action
        elif "time" in ai.text:
            res = ai.action_time()
        ## responding politely
        elif any(i in ai.text for i in ['thank', 'thanks']):
            res = np.random.choice(["You're welcome!", "Anytime!", "No problem!", "Cool!", "I'm here if you need me!"])
        elif any(i in ai.text for i in ["exit", "close"]):
            res = np.random.choice(["Bye!", "Have a good day!", "Hope to see you again.", "See you soon!"])
            ex = False
        else:
            if ai.text == "ERROR":
                res = "Sorry, can you repeat that?"
            else:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("assistant: ")+11:].strip()
        ChatBot.text_to_speech(res)
    print("-------Closing down Neha----------")

import pyttsx3
import speech_recognition as sr
import datetime


engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
#print(voices[0].id)
engine.setProperty('voice',voices[0].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishme():
    hour=int(datetime.datetime.now().hour)
    
    if hour>=0 and hour<12:
        speak("good morning !")
    
    elif hour>=12 and hour<18:
        speak(" good afternoon! ,mam")

    elif hour>=18 and hour<22:
        speak(" a very good evening!")
    else:
        speak("its late mam, you may have a  good sleep , good night")

def takecommand():
    print('jdfheriuhej')
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold=1
        audio=r.listen(source)
    print("hdjfiurhf")
    

    try:
        print("identifying....")
        query=r.recognize_google(audio,language="en-in")
        print(f"you said: {query}")
    except Exception :
        print(" could not understand say that again please")
        return "none"
    return query



if __name__=="__main__":
    wishme()
    speak("i am jarvis, how may i help you? ")
     
    takecommand() 
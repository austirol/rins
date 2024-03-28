from gtts import gTTS

speak = gTTS(text="Hello dear traveller", lang="en", slow=False) 
speak.save("captured_voice.mp3") 

# import required module
from playsound import playsound
 
# for playing note.wav file
playsound('mojca.m4a')
print('playing sound using  playsound')

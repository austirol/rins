import speech_recognition as sr

# List of colors to recognize
colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray"]

def recognize_colors_from_speech():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Use the microphone as the source for input
    with sr.Microphone() as source:
        print("Please say a sentence with two color names:")
        
        # Adjust for ambient noise and record audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        try:
            # Recognize speech using Google Web Speech API
            speech_text = recognizer.recognize_google(audio).lower()
            print("You said:", speech_text)
            
            # Check for the recognized colors in the speech text
            recognized_colors = [color for color in colors if color in speech_text]
            
            if len(recognized_colors) >= 2:
                print(f"Recognized colors: {recognized_colors[:2]}")
                return recognized_colors[:2]
            elif len(recognized_colors) == 1:
                print(f"Only one color recognized: {recognized_colors[0]}")
            else:
                print("No recognized colors in the speech.")
        
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Run the color recognition function
recognize_colors_from_speech()

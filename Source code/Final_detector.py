from PIL import Image, ImageTk   # Importing necessary libraries
import mediapipe as mp
import tkinter as tk
import numpy as np
import threading
import enchant
import pyttsx3
import pickle
import time
import cv2
import os


# Load model
model_dict = pickle.load(open('D:/Sign Language Convertor/Model/model_Final.p', 'rb'))
model = model_dict['model']

# Dictionary mapping numerical labels to characters
labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36:  'Next'
}

# Define custom word list with names
custom_words = {
    'ANIKET', 'SHARMA', 'VISHAL', 'DIVYANSH', 'GUPTA', 'MANYA', 'AYUSH', 'SONIKA' # Add more custom words here
}

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)


# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 2)  # Volume level


class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.iconbitmap('D:/Sign Language Convertor/Images/logo.ico')
        self.vid = cv2.VideoCapture(0)

        # Define the canvas for camera feed
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # can set default values also
        self.canvas.place(x=10, y=20)  # Position the canvas on the left side
        
        # Load and resize the image
        image_dir = 'D:/Sign Language Convertor/Images'  # Change this to your image directory
        img_path = os.path.join(image_dir, 'symbol.png') 
        img = Image.open(img_path)
        img = img.resize((500, 482))
        self.img_tk = ImageTk.PhotoImage(img)
        
        # Label to display the image
        self.image_label = tk.Label(window, image=self.img_tk)
        self.image_label.place(x=670, y=18)
        
        # Labels for prediction
        self.predicted_symbol_label = tk.Label(window, text="Prediction", font=("Helvetica", 20, "bold"))
        self.predicted_symbol_label.place(x=10, y=515)

        self.predicted_word_label = tk.Label(window, text="Word", font=("Helvetica", 20, "bold"), wraplength=700)
        self.predicted_word_label.place(x=10, y=575)

        self.predicted_sentence_label = tk.Label(window, text="Sentence", font=("Helvetica", 20, "bold"), wraplength=700)
        self.predicted_sentence_label.place(x=10, y=635)

        self.prediction_active = False  # Flag to control prediction

        # Buttons for starting/stopping prediction and audio output

        self.start_button = tk.Button(window, text="START PREDICTION", width=20, height=2, command=self.start_prediction, font=("Helvetica", 8, "bold"))
        self.start_button.place(x=775, y=520)

        self.stop_button = tk.Button(window, text="STOP PREDICTION", width=20, height=2, command=self.stop_prediction, font=("Helvetica", 8, "bold"))
        self.stop_button.place(x=935, y=520)

        self.audio_button = tk.Button(window, text="AUDIO OUTPUT", width=30, height=2, command=self.audio_output, font=("Helvetica", 12, "bold"))
        self.audio_button.place(x=775, y=570)

        self.start_button = tk.Button(window, text="CLEAR", width=20, height=2, command=self.clear_output, font=("Helvetica", 8, "bold"))
        self.start_button.place(x=775, y=632)

        self.reset_button = tk.Button(window, text="RESET", width=20, height=2, command=self.reset_output, font=("Helvetica", 8, "bold"))
        self.reset_button.place(x=935, y=632)


        # Label for predicted character
        self.predict_label = tk.Label(window, text="")
        self.predict_label.pack()
        
        self.predicted_character = ""  # Initialize predicted character
        self.word = ""  # Initialize word variable
        self.sentence = ""  # Initialize sentence variable
        self.last_symbol_time = time.time()+5  # Initialize last symbol time
        self.prev_symbol = ""  # Initialize previous symbol

        

        self.update()
        self.window.geometry("1185x700")  # Set the window size here
        self.window.mainloop()
        

    def clear_output(self):
        # Remove the last character from the word
        if self.word:
            self.word = self.word[:-1]
        # Update the labels to reflect the changes
        predicted_word = self.word
        predicted_sentence = self.sentence
        self.predicted_word_label.config(text=f"Word: {predicted_word}")
        self.predicted_sentence_label.config(text=f"Sentence: {predicted_sentence}")

    def write_output_to_file(self, output_text):
        output_file = 'D:/Sign Language Convertor/Output/output.txt'
        with open(output_file, "w") as f:
            f.write(output_text.strip() + "\n")

    def start_prediction(self):
        self.prediction_active = True

    def stop_prediction(self):
        self.prediction_active = False
        self.write_output_to_file(self.sentence)

    def audio_output(self):
        self.prediction_active = False
        output_file = 'D:/Sign Language Convertor/Output/output.txt'
        self.write_output_to_file(self.sentence)

        def play_audio():
            with open(output_file, "r") as file:
                output_text = file.read()
            if output_text:
                engine.say(output_text)
                engine.runAndWait()
            else:
                print("No text available to generate audio.")

        audio_thread = threading.Thread(target=play_audio)
        audio_thread.start()
    
    def reset_output(self):
        # Reset both word and sentence to empty strings
        self.word = ""
        self.sentence = ""
        # Update the labels to reflect the changes
        self.predicted_word_label.config(text="Word:")
        self.predicted_sentence_label.config(text="Sentence:")

    def correct_sentence(self, user_input):
        # Create an English dictionary object
        english_dict = enchant.Dict("en_US")
        corrected_words = []
        for word in user_input.split():
            # Convert word to uppercase
            upper_word = word.upper()
            if upper_word in custom_words:
                # If match found in custom_words, convert to lowercase and append
                corrected_words.append(word)
            elif english_dict.check(word):
                corrected_words.append(word)
            else:
                suggestions = english_dict.suggest(word)
                if suggestions:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
        corrected_sentence = ' '.join(corrected_words)

        return corrected_sentence


    def update(self):
        ret, frame = self.vid.read()

        if ret:
            # frame = cv2.resize(frame, (700, 400))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            
            if results.multi_hand_landmarks and self.prediction_active:

                # Draw the hand annotations/landmark on the image.
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Double the number of features by adding the square of each feature
                    data_aux_squared = [x**2 for x in data_aux]
                    data_aux.extend(data_aux_squared)

                    prediction = model.predict([np.asarray(data_aux)])

                # Convert prediction to numerical label if it's a string
                if isinstance(prediction[0], str):
                    self.predicted_character = labels_dict[int(prediction[0])]
                else:
                    self.predicted_character = int(prediction[0])

                # # Draw rectangle around hand
                # x1 = int(min(x_) * frame.shape[1]) - 10
                # y1 = int(min(y_) * frame.shape[0]) - 10
                # x2 = int(max(x_) * frame.shape[1]) - 10
                # y2 = int(max(y_) * frame.shape[0]) - 10
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # # Display predicted character
                # cv2.putText(frame, str(self.predicted_character), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # # Display predicted character
                # cv2.putText(frame, str(self.predicted_character), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # Display predicted string with label names
                predicted_word = self.word  # You need to define this variable according to your implementation
                predicted_sentence = self.sentence  # You need to define this variable according to your implementation

                self.predicted_symbol_label.config(text=f"Prediction: {self.predicted_character}")  # Use self.predicted_character here
                self.predicted_word_label.config(text=f"Word: {predicted_word}")
                self.predicted_sentence_label.config(text=f"Sentence: {predicted_sentence}")

                # Check if symbol is not blank
                if self.predicted_character != "Blank":

                    if self.predicted_character == "Next":
                        if self.sentence and self.sentence[-1] != " ":
                            self.sentence += " "
                        else:
                            self.sentence += self.correct_sentence(self.word)

                        self.word = ""
                        self.last_symbol_time = time.time()  # Update last symbol time
                        print("reset_next")


                    # If it's been 3 seconds since last symbol change and previous symbol is the same as current symbol, add symbol to word
                    if time.time() - self.last_symbol_time >= 3:
                        if self.prev_symbol == self.predicted_character:
                            self.word += str(self.predicted_character)
                            self.last_symbol_time = time.time()  # Update last symbol time
                            print("reset_symbol_match")
                        else:
                            # Reset last_symbol_time if the predicted character changes
                            self.last_symbol_time = time.time()
                            print("reset_change")



            else:
                self.predicted_character = "Blank"  # No hand detected, set the predicted character to blank
                self.predicted_symbol_label.config(text=f"Prediction: {self.predicted_character}")  # Display "Blank" in symbol label
                self.last_symbol_time = time.time()  # Update last symbol time
                print("reset_blank")

            # Update previous symbol
            self.prev_symbol = self.predicted_character    

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.photo = photo  # Keep a reference to avoid garbage collection
        

        self.window.after(10, self.update)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
if __name__ == '__main__':
    App = Application(tk.Tk(), "SIGN LANGUAGE DETECTOR")

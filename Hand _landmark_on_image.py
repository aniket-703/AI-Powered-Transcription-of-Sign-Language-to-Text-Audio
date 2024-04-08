import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# Define mp_hands and mp_drawing globally
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to detect hand landmarks using MediaPipe Hands
def detect_hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3) as hands:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(image_rgb)
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
    
    return image


# Function to select an image using a dialog box
def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def main():
    while True:
        # Select an image
        image_path = select_image()
        if not image_path:
            print("Error: No image selected.")
            break

        # Read the selected image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to read the image.")
            break

        # Detect hand landmarks
        image_with_landmarks = detect_hand_landmarks(image)

        # Display the image with hand landmarks
        cv2.imshow("Image with Hand Landmarks", image_with_landmarks)
        
        # Wait for user input to continue or exit
        key = cv2.waitKey(0)
        if key == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

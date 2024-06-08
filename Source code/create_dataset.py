import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Define the directory containing the image data
DATA_DIR = 'D:/Sign Language Convertor/Data'

# Initialize lists to store data and labels
data = []
labels = []

# Iterate through each directory and image in the data directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        # Temporary list to store hand landmark data for each image
        data_aux = []   

        # List to store x & y coordinates of hand landmarks
        x_ = []  
        y_ = []  

        # Read the image and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        # Check if hand landmarks were detected in the image & Iterate through each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Extract x and y coordinates of each hand landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # Store x and y coordinates in separate lists
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates by subtracting the minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the hand landmark data and corresponding label to the lists
            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels to a pickle file
try:
    with open('D:/Sign Language Convertor/Model/data10.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data saved successfully.")
except Exception as e:
    print("Error occurred while saving the data:", e)

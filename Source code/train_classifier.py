import pickle 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Loading the preprocessed data from a pickle file
data_dict = pickle.load(open('D:/Sign Language Convertor/Model/data10.pickle', 'rb'))

# Finding the maximum length of sequences in the data
max_length = max(len(seq) for seq in data_dict['data'])

# Pading sequences with zeros to make them all the same length
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

# Converting the data and labels to NumPy arrays
data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Normalize/Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Creating a Random Forest classifier model
model = RandomForestClassifier()

# Training the model using the training data
model.fit(x_train, y_train)

# Making predictions on the test data
y_predict = model.predict(x_test)

# Calculating the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Printing the accuracy of the model
print('{}% of samples were classified correctly!'.format(score * 100))

# Saving the model
with open('D:/Sign Language Convertor/Model/model10.p', 'wb') as f:
    pickle.dump({'model': model}, f)
f.close()

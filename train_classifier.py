import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('D:/sign-language-detector-python-master/data.pickle', 'rb'))

# Find the maximum length of sequences in the data
max_length = max(len(seq) for seq in data_dict['data'])

# Pad sequences with zeros to make them all the same length
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('D:/sign-language-detector-python-master/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
f.close()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

#TODO: THE HIGHEST SCORE WAS WITHOUT BATCH NORMALIZER, WITH ALL NON-CATEGORIAL FEATURES (INPUT SHAPE 30 AND WITHOUT THE SECOND LAYER, AND BATCH SIZE 128).

directory='data/'

# Load the data
labels = pd.read_csv(directory + 'train_labels.csv')
features = pd.read_csv(directory + 'train_values.csv')

labels.set_index('building_id', inplace=True)
features.set_index('building_id', inplace=True)

# Data wrangling: FEATURES
# Remove variables that might not be useful
features.drop(['plan_configuration', 'position'], axis=1, inplace=True)

# Select only numerical features
num_features = features.select_dtypes(include=np.number)
# Switch to array
num_features = num_features.values
# Scale features
num_features = MinMaxScaler().fit_transform(num_features)

# Reformat the labels so it goes from 0 to 2 instead than 1-3
labels['damage_grade'] = labels['damage_grade'] - 1
# One hot encoding of labels plus convert to array.
labels = np_utils.to_categorical(labels)


# Split train and test (dev)
X_train, X_test, y_train, y_test = train_test_split(num_features, labels, test_size=0.15, random_state=33)


# DEEP LEARNING MODEL ---
model = Sequential()
model.add(Dense(30, activation= 'relu', input_shape=(30,)))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation= 'softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=10, batch_size = 128, verbose=2, validation_data=(X_test, y_test))


# PREDICTIONS
# Apply the results to the test:
# Data wrangling
test_values = pd.read_csv(directory + 'test_values.csv')
test_values.set_index('building_id', inplace=True)

# Data wrangling: FEATURES
# Remove variables that might not be useful
test_values.drop(['plan_configuration', 'position'], axis=1, inplace=True)
# Select only numerical values
num_features_test = test_values.select_dtypes(include=np.number)
# Convert to array
num_features_test = num_features_test.values
# Scale features
num_features_test = MinMaxScaler().fit_transform(num_features_test)

# Predict on the test set
results_test = model.predict(num_features_test)

# Get the highest value as the predicted class
results_test = pd.DataFrame(results_test)
results_test['damage_grade'] = results_test.values.dot(results_test.columns)
results_test['damage_grade'] = results_test['damage_grade'].round()

building_id = test_values.index
building_id = list(building_id)

results_test['building_id'] = building_id
#results_test.rename(columns={0:'damage_grade'}, inplace=True)
results_test = results_test[['building_id', 'damage_grade']]
results_test['damage_grade'] = results_test['damage_grade'].astype(int)

# Again, reformat the damage grade to be 1-3 instead of 0-2
results_test['damage_grade'] = results_test['damage_grade'] + 1

results_test.to_csv('submission_DL.csv', index=False)

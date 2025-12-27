import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import joblib

NUM_SAMPLES = 1000
NUM_FEATURES = 4
NUM_CLASSES = 4

np.random.seed(42)
tf.random.set_seed(42)

X_raw = np.random.rand(NUM_SAMPLES, NUM_FEATURES) * np.array([80, 300, 1000, 200])
y_indices = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_seq = np.expand_dims(X_scaled, axis=1)

y_onehot = tf.keras.utils.to_categorical(y_indices, num_classes=NUM_CLASSES)

model = Sequential()
model.add(LSTM(32, input_shape=(X_seq.shape[1], X_seq.shape[2])))
model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

model.fit(X_seq, y_onehot, epochs=20, batch_size=16, verbose=1, validation_split=0.2)

model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved.")

# Import necessary libraries
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GlobalMaxPool1D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
import librosa
import pywt
import pickle
from sklearn.metrics import classification_report
import joblib

# Define functions for computing wavelet features
def compute_wavelet_features(X, a, b):
    # Define a few parameters
    wavelet = 'morl' # wavelet type: morlet
    sr = 9000 # sampling frequency: 8KHz
    widths = np.arange(1, 256) # scales for morlet wavelet 
    dt = 1/sr # timestep difference

    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    
    # Create a filter to select frequencies between 80Hz and 5KHz
    upper_hss = ([x for x in range(len(widths)) if frequencies[x] > b])[-1]
    lower_hss = ([x for x in range(len(widths)) if frequencies[x] < a])[0]
    
   
    
    widths_hss = widths[upper_hss:lower_hss]
    # Select scales in this frequency range

    # Compute continuous wavelet transform of the audio numpy array
    wavelet_coeffs_hss, freqs_hss = pywt.cwt(X, widths_hss, wavelet = wavelet, sampling_period=dt)
    
    # print(wavelet_coeffs.shape)
    # sys.exit(1)

    # Split the coefficients into frames of length 800
    start = 0
    
    end_hss = wavelet_coeffs_hss.shape[1]
    
    
    frames_hss = []
    
    frame_size = 400
    #count = 0

    while start+frame_size <= end_hss-1 :

        f_hss = wavelet_coeffs_hss[:,start:start+frame_size]
       

        # Total samples in a frame will not be a multiple of 800 everytime. If the last frame length is less than 800, we can skip it.
        assert f_hss.shape[1] == frame_size
        # assert frame lengths are equal to the frame_size parameter

        frames_hss.append(f_hss)
        
        start += frame_size


    # Convert frames to numpy array
    frames_hss = np.array(frames_hss)
    frames_hss = frames_hss.reshape((len(frames_hss), wavelet_coeffs_hss.shape[0], frame_size))
    
    

    return frames_hss

# Define a function for creating the deep learning model
def create_model(row, col):
   n_filters = 32
   filter_width = 3
   dilation_rates = [2**i for i in range(6)] * 2

   # define an input history series and pass it through a stack of dilated causal convolution blocks
   history_seq = Input(shape=(row, col))
   x = history_seq

   #skips = []
   #count = 0
   # x = GaussianNoise(0.01)(x)
   for dilation_rate in dilation_rates:

       # preprocessing - equivalent to time-distributed dense
       
       # filter
       x = Conv1D(filters=n_filters,
                   kernel_size=filter_width, 
                   padding='causal',
                   dilation_rate=dilation_rate, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
   
       x = BatchNormalization()(x)
       x = Activation('relu')(x)

   out = Conv1D(16, 3, padding='same', kernel_initializer= 'random_normal', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
   out = BatchNormalization()(out)
   out = Activation('relu')(out)
   out = GlobalMaxPool1D()(out)

   out = Dense(3, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(out)
   out = Activation('softmax')(out)

   model = Model(history_seq, out)

   model.compile(loss='categorical_crossentropy', optimizer='adam')

   return model

# Load the training data
training_data = np.load(os.getcwd() + "/training_features.npz", allow_pickle=True)
X_train = training_data['a']
y_train = training_data['b']

# Transpose the data to have the correct format
X_train = X_train.transpose(0, 2, 1)

# Convert class labels to categorical vectors
y_train = to_categorical(y_train)

# Standardize the data
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

# Use 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)

count = 0

# Train the model using cross-validation
for train_indices, test_indices in kfold.split(X_train, np.argmax(y_train, axis=-1)):
    print("K Fold Step ", count)

    model = create_model(X_train.shape[1], X_train.shape[2])
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train[train_indices], y_train[train_indices], validation_data=(X_train[test_indices], y_train[test_indices]), batch_size=128, epochs=80, verbose=2)

    # Save the trained model
    model.save(os.getcwd() + "/speaker_classifier_fold{}.h5".format(count))

    count += 1

    scores = model.evaluate(X_val, y_val, verbose=0)
    print("Validation Loss: {:.4f}".format(scores))

# Test the model
model = load_model(os.getcwd() + "/hss_lss_data.h5", compile=False)  # Load one of the saved models

# Load the standard deviation and mean
f = open(os.getcwd() + '/hss_mean_std.pkl', 'rb')
mean, std, _ = pickle.load(f)
f.close()

testing_data = np.load(os.getcwd() + "/testing_features.npz", allow_pickle=True)
X_test = testing_data['a']
y_test = testing_data['b']
ind = testing_data['c']
unq_ind = np.unique(ind)

X_test = X_test.astype(np.float32)
X_test = X_test.transpose(0, 2, 1)
X_test = (X_test - mean) / std

# Predict
ypred = model.predict(X_test)
ypred = np.argmax(ypred, axis=-1)
ypred = ypred.flatten()

new_pred = []
new_truth = []

# Find unique ids and assign class based on majority vote from all the frames
for i in range(len(unq_ind)):
    curr = unq_ind[i]
    indices, = np.where(ind == curr)
    t = y_test[indices]
    t = t[0]

    p = ypred[indices]

    un, fr = np.unique(p, return_counts=True)
    new_pred.append(un[np.argmax(fr)])

    print("Truth ", t, " Pred ", un, fr)

    new_truth.append(t)

new_truth = np.array(new_truth)
new_pred = np.array(new_pred)

# Print classification report
rep = classification_report(new_truth, new_pred, target_names=['hss', 'lss'])
print(rep)

# Save the model and mean/std deviation for future use
joblib.dump(model, '.pkl')
joblib.dump((mean, std), 'hss_mean_std.pkl')



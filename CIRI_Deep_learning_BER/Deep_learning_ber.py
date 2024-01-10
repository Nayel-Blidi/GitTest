
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.special import erfc

G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

length = 100
n_bits = 8*length 
EbN0_dB = np.linspace(-2, 6, 9)
mod_order = 2 
# Noise variance
EbN0 = 10**(EbN0_dB/10)  # Eb/N0 in linear scale
sigma = np.sqrt(1/EbN0) # Base probability for bit 0
bits = np.random.randint(2, size=n_bits)

import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Dense, Lambda
from keras import backend as K
from keras.models import Sequential

def modulateBPSK(x):
    return -2*x +1;

def reshape(signal):
    return signal @ G

def addNoise(x, sigma):
    w = sigma*K.random_normal(K.shape(x), mean=0.0)
    return x + w

def ber(y_true, y_pred):
    return K.mean(K.not_equal(y_true, K.round(y_pred)))

def return_output_shape(input_shape):  
    return input_shape

def compose_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    return model

def log_likelihood_ratio(x, sigma):
    return 2*x/np.float32(sigma**2)

def errors(y_true, y_pred):
    return K.sum(K.not_equal(y_true, K.round(y_pred)))

def theoretical_BER(EbN0):
    return 0.5*erfc(np.sqrt(EbN0))

k = 8                       # number of information bits
N = 16                      # code length
train_SNR_Eb = 1            # training-Eb/No

nb_epoch = 2**16            # number of learning epochs
code = 'polar'              # type of code ('random' or 'polar')
design = [128, 64, 32]      # each list entry defines the number of nodes in a layer
batch_size = 256            # size of batches for calculation the gradient
LLR = False                 # 'True' enables the log-likelihood-ratio layer
optimizer = 'adam'           
loss = 'mse'                # or 'binary_crossentropy'

combinations_8 = np.zeros((256, 8))
for i in range(256):
    binary = bin(i)[2:].zfill(8)
    bits = [int(b) for b in binary]
    combinations_8[i, :] = bits

combinations_16 = reshape(combinations_8)
Combinations_16 = np.mod(combinations_16, 2)
combination = modulateBPSK(Combinations_16)

sigma = np.sqrt(1/10**(1/10))   # best sigma is for Eb/N0(dB) = 1
y_train = combinations_8        # All 8-bits combinations
x_train = Combinations_16       # Pollar code of the 8-bits combinations

# %% MODEL FUNCTION CALLING

def model_definition(x_train, y_train, sigma, nb_epochs=2**12, batch_size=256, verbose=2):
    
    input_model = Sequential()
    noise_model = Sequential()
    decoder_model = Sequential()
    global_model = Sequential()
    
    # Input layer
    input_layer = keras.layers.Lambda(modulateBPSK, input_dim=16, name="symbols_input")
    input_model.add(input_layer)
    input_model.compile(optimizer=optimizer, loss=loss)
    
    # Noise layer
    noise_layer = keras.layers.Lambda(addNoise, arguments={'sigma':sigma}, input_dim=16, name="noise")
    noise_model.add(noise_layer)
    noise_model.compile(optimizer=optimizer, loss=loss)
    
    # Neurons layers
    decoder_layers = []
    decoder_layers.append(Dense(128, input_dim=16, activation='relu'))
    decoder_layers.append(Dense(64, activation='relu'))
    decoder_layers.append(Dense(32, activation='relu'))
    decoder_layers.append(Dense(8, activation='sigmoid'))
    for layer in decoder_layers:
        decoder_model.add(layer)
    decoder_model.compile(optimizer=optimizer, loss=loss, metrics=[errors])
    
    # Global model 
    model_layers = [input_layer, noise_layer] + decoder_layers
    for layer in model_layers:
        global_model.add(layer)
    global_model.compile(optimizer=optimizer, loss=loss, metrics=[ber])
    
    if verbose >= 1:
        global_model.summary()
        
    history = global_model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=verbose)

    return global_model, decoder_model, history

# %% 
model, decoder, history = model_definition(x_train, y_train, sigma, nb_epochs=2**18, batch_size=256, verbose=0)

# %% MODEL TESTING

def model_evaluation(G, model, n_words = 1000, sigma=0.89, itterations=100):
    
    ber_score = 0
    for itteration in tqdm(range(itterations)):
        message = np.random.randint(2, size=(n_words, 8))
        message_polar = modulateBPSK(np.mod(reshape(message), 2))
        message_polar = message_polar + sigma * (np.random.standard_normal(size=(n_words, 16)))
        
        prediction = np.round(model.predict(message_polar, verbose=0))
        nb_errors = np.sum(np.not_equal(prediction, message))
        ber = nb_errors/(n_words*8)

        ber_score += ber

    return ber_score/itterations, message, prediction, message_polar

import os
from scipy import polyfit

def MAP_data(show_plot=0):

    file_location = os.path.realpath(__file__)
    os.chdir(os.path.dirname(file_location))
    
    # Loading MAP values
    result_map = np.loadtxt('results_polar_map_16_8.txt', delimiter=',')
    sigmas_map = result_map[:,0]
    nb_bits_map = result_map[:,1]
    nb_errors_map = result_map[:,2]
    ber_map = nb_errors_map/nb_bits_map
    
    results_map_extrapolated = np.loadtxt("results_polar_map_extapolated.txt", delimiter=',')
    EbN0_dB_extrapolated = results_map_extrapolated[:,0]
    ber_map_extrapolated = results_map_extrapolated[:,1]
    
    EbN0 = 1/(sigmas_map**2)
    EbN0_dB = 10*np.log10(EbN0)
    
    EbN0_dB_new = np.concatenate([EbN0_dB, EbN0_dB_extrapolated], axis=0)
    ber_map_new = np.concatenate([ber_map, ber_map_extrapolated], axis=0)    
    
    coeffs = polyfit(EbN0_dB_new, ber_map_new, 6)
    ber_pred = np.polyval(coeffs, EbN0_dB_new)
    if show_plot != 0:
        plt.semilogy(EbN0_dB_new, ber_pred)
        plt.grid(True, "both")
    
    return EbN0_dB_new, ber_pred

# %%
sigma = np.sqrt(1/10**(1/10))   # best sigma is for Eb/N0(dB) = 1
n_words = 1000

ber_score, message, prediction, message_polar = model_evaluation(G, decoder, n_words, sigma, itterations=100)
EbN0_dB_new, ber_pred = MAP_data()

EbN0_dB = np.linspace(-2, 8, 11)
EbN0 = 10**(EbN0_dB/10)
sigma_test = np.sqrt(1/EbN0)

ber_score_hist = []
for k in range(len(EbN0)):
    ber_score_hist.append(model_evaluation(G, decoder, sigma=sigma_test[k], itterations=100)[0])

plt.semilogy(EbN0_dB, ber_score_hist, label=f"nb_epochs = $2^{{{18}}}$")
plt.semilogy(EbN0_dB_new, ber_pred, "r--", label="MAP")
plt.legend()
plt.xlabel("EbN0_dB")
plt.ylabel("BER")
plt.grid(True, "both")

# %% COMPLETE MODELS TESTING (sigmas_dB) AND BER EVALUATIONS (Eb/N0)

def model_training_and_evaluation(G, n_words, EbN0_dB, training_epochs_power, testing_itteration=10):
    
    EbN0 = 10**(EbN0_dB/10)
    MAP_EbN0, MAP_BER = MAP_data()
    plt.close('all')

    sigma_test = np.sqrt(1/EbN0)
    for EbN0_value in EbN0:
        sigma_training = np.sqrt(1/EbN0_value)
        for pow_epochs in training_epochs_power:
            nb_epochs = 2**pow_epochs
            print(f"\nComputing NN {nb_epochs} epochs, sigma_training={round(sigma_training, 2)}")
            model, decoder, history = model_definition(x_train, y_train, sigma_training, nb_epochs, verbose=0)
            
            ber_score_hist = []
            for k in range(len(EbN0)):
                ber_score_hist.append(model_evaluation(G, decoder, n_words, sigma_test[k], testing_itteration)[0])
        
            plt.semilogy(EbN0_dB, ber_score_hist, label=f"nb_epochs = $2^{{{pow_epochs}}}$")  
            
        plt.semilogy(MAP_EbN0, MAP_BER, 'r--', label="MAP")
        plt.xlabel("EbN0_dB")
        plt.ylabel("BER")
        plt.axis([EbN0_dB[0], EbN0_dB[-1], 10e-6, 1])
        plt.legend(loc="lower left")
        plt.grid(True, which="both")
        plt.title(f"NN BER decoding results, sigma_training={round(sigma_training, 2)}")
        plt.savefig(f"C:/Travail/IPSA/Aero4/Machine Learning/Deep_learning_BER/BERmodel_test_itt={testing_itteration}_sigma_train={round(sigma_training, 2)}.png")
        plt.close('all')

    return None

# %% PLOTS OUTPUT & SAVING
EbN0_dB = np.linspace(0, 7, 8) # Eb/N0 ratio range in dB
n_words = 1000 # Number of testing 8-bits words
#training_epochs_power = [10, 12, 14, 16, 18]
training_epochs_power = [10, 12, 14, 16]
#training_epochs_power = [10, 12]

model_training_and_evaluation(G, n_words, EbN0_dB, training_epochs_power, testing_itteration=100)




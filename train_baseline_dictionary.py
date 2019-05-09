import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from seaborn import set_palette
import pickle
flatui = ["#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
set_palette(flatui)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linewidth'] = 0.25
plt.rcParams['grid.alpha'] = 0.2
plt.style.use('seaborn-talk')

from scipy.signal import stft, istft, get_window
from IPython.display import Audio
from tqdm import tnrange, tqdm_notebook
from dlbeamformer_utilities import compute_steering_vectors_single_frequency,\
    compute_steering_vectors, simulate_multichannel_tf, compute_sinr,\
    compute_mvdr_tf_beamformers, check_distortless_constraint
from dlbeamformers import BaseDLBeamformer
random_seed = 0

import os
from os import listdir
from os.path import join
datapath = "CMU_ARCTIC/cmu_us_bdl_arctic/wav"
train_data_folder = join(datapath, 'train')
test_data_folder = join(datapath, 'test')

from scipy.io import wavfile
from IPython.display import Audio
train_data = []
test_data = []
train_data_filenames = [f for f in listdir(train_data_folder) if os.path.isfile( join(train_data_folder, f))]
test_data_filenames = [f for f in listdir(test_data_folder) if os.path.isfile( join(test_data_folder, f))]

for i_train_data_filename in range(len(train_data_filenames)):
    f_path = join(train_data_folder, train_data_filenames[i_train_data_filename])
    if f_path.endswith('.wav'):
        sampling_frequency, train_data_example = wavfile.read(f_path)
    train_data.append(train_data_example)
    
for i_test_data_filename in range(len(test_data_filenames)):
    f_path = join(test_data_folder, test_data_filenames[i_test_data_filename])
    if f_path.endswith('.wav'):
        sampling_frequency, test_data_example = wavfile.read(f_path)
    test_data.append(test_data_example)
    
# Microphone positions
pos_x = np.arange(-0.8, 0.8+1e-6, 0.2)
n_mics = len(pos_x)
pos_y = np.zeros(n_mics)
pos_z = np.zeros(n_mics)
array_geometry = np.row_stack((pos_x, pos_y, pos_z))


from configparser import ConfigParser
config = ConfigParser()
config.read('config.INI');
params = config['PARAMS']
sampling_frequency = int(params['sampling_frequency'])
n_samples_per_frame = int(params['n_samples_per_frame'])
n_fft_bins = (int) (n_samples_per_frame / 2) 
hop_size = (int) (n_samples_per_frame / 2)
stft_window_name = params['stft_window_name']
stft_window = get_window("hann", n_samples_per_frame)
stft_params = {
    "n_samples_per_frame": n_samples_per_frame,
    "n_fft_bins": n_fft_bins,
    "hop_size": hop_size,
    "window": stft_window
}

# Source angles
theta_s = np.array([-10]) # [degree]
phi_s = np.array([0]) # [degree]

# Angle grids
theta_grid = np.arange(-90, 90+1e-6, 0.1) # [degree]
phi_grid = np.array([0]) # [degree]

# Steering vectors
steering_vectors = compute_steering_vectors(array_geometry, sampling_frequency=sampling_frequency, 
                                    n_fft=n_fft_bins, theta_grid=theta_grid, phi_grid=phi_grid)
source_steering_vectors = compute_steering_vectors(array_geometry, sampling_frequency, n_fft_bins, 
    np.array([theta_s[0]]), np.array([phi_s[0]]))


np.random.seed(random_seed)
n_interference_list = [1, 2]

azimuth_step = 30
training_thetas = list(np.arange(-90, 90, azimuth_step))

training_phis = [0]

import itertools
training_interference_data = []
training_noise_interference_data = []
np.random.seed(random_seed)

for i_n_interference in tqdm_notebook(range(len(n_interference_list)), desc="Interference number"):
    n_interferences = n_interference_list[i_n_interference]
    interferences_params = []
    for i_interference in range(n_interferences):
        interference_params = list(itertools.product(*[training_thetas, training_phis]))
        interferences_params.append(interference_params)
    interferences_param_sets = list(itertools.product(*interferences_params))

    for i_param_set in tqdm_notebook(range(len(interferences_param_sets)), desc="Parameter set"):    
        param_set = interferences_param_sets[i_param_set]
        n_training_samples = 5
        for i_training_sample in range(n_training_samples):
            interference_signals = []
            for i_interference in range(len(param_set)):
                interference_signal = train_data[np.random.choice(len(train_data))]
                interference_signals.append(interference_signal)                
            interference_n_samples = min([len(signal) for signal in interference_signals])
            
            interference_tf_multichannel_list = []
            for i_interference in range(len(param_set)):
                interference_signals[i_interference] = (interference_signals[i_interference])[0:interference_n_samples]
                interference_theta, interference_phi = param_set[i_interference]
                interference_theta += 2*np.random.uniform()
                interference_tf_multichannel = simulate_multichannel_tf(array_geometry, interference_signal, 
                        np.array([interference_theta]), np.array([interference_phi]),
                        sampling_frequency, stft_params)
                interference_tf_multichannel_list.append(interference_tf_multichannel)
            training_interference_data.append(sum(interference_tf_multichannel_list))
            
dictionary = BaseDLBeamformer(source_steering_vectors[:, 0, 0, :])
dictionary.fit(training_interference_data);


dict_filename = "baseline_dl_azimuth_step_{}_trainning_samples_{}.pkl".format(
    azimuth_step, n_training_samples
)
train_models_path = "trained_models"
dict_filepath = os.path.join(train_models_path, dict_filename)
with open(dict_filepath, 'wb') as output:
    pickle.dump(dictionary, output, pickle.HIGHEST_PROTOCOL)
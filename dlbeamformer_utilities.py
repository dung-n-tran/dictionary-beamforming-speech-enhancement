import numpy as np
from scipy.signal import stft
SOUND_SPEED = 340 # [m/s]
# Steering vectors
def compute_steering_vectors_single_frequency(array_geometry, frequency, theta_grid, phi_grid):
    # wave number
    k = 2*np.pi*frequency/SOUND_SPEED

    n_mics = len(array_geometry[0])
    theta_grid = theta_grid * np.pi/180 # [degree] to [radian]
    phi_grid = phi_grid * np.pi/180 # [degree] to [radian]
    
    u = np.sin(theta_grid.reshape(-1, 1)).dot(np.cos(phi_grid).reshape(1, -1))
    v = np.sin(theta_grid.reshape(-1, 1)).dot(np.sin(phi_grid).reshape(1, -1))
    w = np.tile(np.cos(theta_grid.reshape(-1, 1)), (1, phi_grid.shape[0]))

    x = u.reshape(u.shape[0], u.shape[1], 1)*array_geometry[0].reshape(1, 1, n_mics)
    y = v.reshape(v.shape[0], v.shape[1], 1)*array_geometry[1].reshape(1, 1, n_mics)
    z = w.reshape(w.shape[0], w.shape[1], 1)*array_geometry[2].reshape(1, 1, n_mics)

    return np.exp( -1j*k*(x + y + z))

def compute_steering_vectors(array_geometry, sampling_frequency, n_fft, theta_grid, phi_grid):
    n_thetas = len(theta_grid)
    n_phis = len(phi_grid)
    n_mics = len(array_geometry[0])
    steering_vectors = np.zeros((n_fft, n_thetas, n_phis, n_mics), dtype=np.complex64)
    for i_fft in range(n_fft):
        frequency = (i_fft / n_fft) * (sampling_frequency)
        steering_vectors[i_fft] = compute_steering_vectors_single_frequency(array_geometry, frequency, theta_grid, phi_grid)
        
    return steering_vectors

def compute_sinr_2(source_tf_multichannel, interference_tf_multichannel):
        source_power = 0
        interference_power = 0
        n_fft_bins = source_tf_multichannel.shape[0]
        for i_f in range(n_fft_bins):
            source_power += np.trace(source_stft_multichannel[i_f].dot(source_stft_multichannel[i_f].transpose().conjugate()))
            interference_power += np.trace(interference_stft_multichannel[i_f].dot(interference_stft_multichannel[i_f].transpose().conjugate()))
        return 10*np.log10(np.abs(source_power/interference_power))
    
def compute_sinr(source_tf_multichannel, interference_tf_multichannel, weights=None):
    n_fft_bins, n_mics, _ = source_tf_multichannel.shape
    source_power = 0
    interference_power = 0
    if weights is not None:
        for i_f in range(n_fft_bins):
            source_power += weights[i_f].reshape(n_mics, 1).transpose().conjugate().dot(
                source_tf_multichannel[i_f].dot(
                source_tf_multichannel[i_f].transpose().conjugate())).dot(
                weights[i_f].reshape(n_mics, 1))
            interference_power += weights[i_f].transpose().conjugate().dot(
                interference_tf_multichannel[i_f].dot(
                interference_tf_multichannel[i_f].transpose().conjugate())).dot(
                weights[i_f])
    else:
        for i_f in range(n_fft_bins):
            source_power += np.trace(source_tf_multichannel[i_f].dot(source_tf_multichannel[i_f].transpose().conjugate()))
            interference_power += np.trace(interference_tf_multichannel[i_f].dot(interference_tf_multichannel[i_f].transpose().conjugate()))
    return 10*np.log10(np.abs(source_power/interference_power))
    
def compute_mvdr_tf_beamformers(source_steering_vectors, tf_frames_multichannel):
    n_fft_bins, n_mics = source_steering_vectors.shape
    mvdr_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        R = tf_frames_multichannel[i_fft_bin].dot(tf_frames_multichannel[i_fft_bin].transpose().conjugate()) + np.identity(n_mics)
        invR = np.linalg.inv(R)
        normalization_factor = source_steering_vectors[i_fft_bin, :].transpose().conjugate().dot(invR).dot(source_steering_vectors[i_fft_bin, :])
        mvdr_tf_beamformers[i_fft_bin] = invR.dot(source_steering_vectors[i_fft_bin, :]) / normalization_factor
    return mvdr_tf_beamformers

def simulate_multichannel_tf(array_geometry, signal, theta, phi, sampling_frequency, stft_params):
    n_mics = len(array_geometry[0])
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    n_fft_bins = stft_params["n_fft_bins"]
    hop_size = stft_params["hop_size"]
    stft_window = stft_params["window"]
    steering_vector = ( compute_steering_vectors(array_geometry, sampling_frequency, n_fft_bins, theta, phi) )[:, 0, 0, :]
    _, _, tf_frames = stft(signal.reshape(-1), fs=sampling_frequency, window=stft_window,
                             nperseg=n_samples_per_frame, noverlap=n_samples_per_frame-hop_size,
                             nfft=n_samples_per_frame, padded=True)
    tf_frames = tf_frames[:-1, 1:-1]
    tf_frames_multichannel = steering_vector.reshape(n_fft_bins, n_mics, 1)\
                                * tf_frames.reshape(tf_frames.shape[0], 1, tf_frames.shape[1])
    return tf_frames_multichannel

def check_distortless_constraint(weight, source_steering_vector):
    assert(np.abs(weight.transpose().conjugate().dot(source_steering_vector)) - 1 < 1e-9)
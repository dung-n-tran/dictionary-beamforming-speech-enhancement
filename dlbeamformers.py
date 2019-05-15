import numpy as np
from dlbeamformer_utilities import compute_mvdr_tf_beamformers, check_distortless_constraint, compute_steering_vectors
from tqdm import tnrange, tqdm

class BaseDLBeamformer(object):
    def __init__(self, vs, bf_type="MVDR"):
        """
        Parameters
        ----------
        vs: Source manifold array vector
        bf_type: Type of beamformer
        """
        self.vs = vs
        self.bf_type = bf_type
        self.weights_ = None
        
    def _compute_weights(self, training_data):
        n_training_samples = len(training_data)
        n_fft_bins, n_mics, _ = training_data[0].shape
        D = np.zeros((n_fft_bins, n_mics, n_training_samples), dtype=complex)
        for i_training_sample in tqdm(range(n_training_samples), desc="Training sample"):
            tf_frames_multichannel = training_data[i_training_sample]
            if self.bf_type == "MVDR":
                w = compute_mvdr_tf_beamformers(self.vs, tf_frames_multichannel)
#                 check_distortless_constraint(w, self.vs)
            D[:, :, i_training_sample] = w
            
        return D

    def _initialize(self, X):
        pass

    def _choose_weights(self, x):
        n_dictionary_atoms = self.weights_.shape[2]
        min_ave_energy = np.inf
        optimal_weight_index = None
        for i_dictionary_atom in range(n_dictionary_atoms):
            w_frequency = self.weights_[:, :, i_dictionary_atom]
            energy = 0
            n_fft_bins = w_frequency.shape[0]
            for i_fft_bin in range(n_fft_bins):
                w = w_frequency[i_fft_bin]
                R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
                energy += np.real(w.transpose().conjugate().dot(R).dot(w))
            ave_energy = energy / n_fft_bins
            if min_ave_energy > ave_energy:
                min_ave_energy = ave_energy
                optimal_weight_index = i_dictionary_atom
        optimal_weight = self.weights_[:, :, optimal_weight_index]
        return optimal_weight, optimal_weight_index
    
    def fit(self, training_data):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._compute_weights(training_data)
        self.weights_ = D
        return self

    def choose_weights(self, x):
        return self._choose_weights(x)
    
class DLBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, bf_type="MVDR"):
        """
        Parameters
        ----------
        array_geometry: 2-D numpy array describing the geometry of the microphone array
        sampling_frequency
        stft_params: Dictionary of STFT transform parameters including
            stft_params["n_samples_per_frame"]
            stft_params["n_fft_bins"]
            stft_params["hop_size"]
            stft_params["window"]
        bf_type: Type of the beamformer
        """
        self.array_geometry = array_geometry
        self.sampling_frequency = sampling_frequency
        self.source_angles = source_angles
        self.stft_params = stft_params
        self.angle_grid = angle_grid
        self.bf_type = bf_type
        self.weights_ = None
        self.source_steering_vectors = self._compute_source_steering_vectors()
        self.steering_vectors = self._compute_steering_vectors()
        
    def _compute_source_steering_vectors(self):
        source_steering_vectors = []
        for i_source_angle, source_angle in enumerate(self.source_angles):
            v = compute_steering_vectors(self.array_geometry, 
                    self.sampling_frequency, self.stft_params["n_fft_bins"], 
                    source_angle["theta"], source_angle["phi"])
            source_steering_vectors.append(v)
        return source_steering_vectors
    
    def _compute_steering_vectors(self):
        return compute_steering_vectors(self.array_geometry,
                    self.sampling_frequency, self.stft_params["n_fft_bins"],
                    self.angle_grid["theta"], self.angle_grid["phi"])
#     def _compute_weights(self, training_data):
#         n_training_samples = len(training_data)
#         n_fft_bins, n_mics, _ = training_data[0].shape
#         D = np.zeros((n_fft_bins, n_mics, n_training_samples), dtype=complex)
#         for i_training_sample in tqdm(range(n_training_samples), desc="Training sample"):
#             tf_frames_multichannel = training_data[i_training_sample]
#             if self.bf_type == "MVDR":
#                 w = compute_mvdr_tf_beamformers(self.vs, tf_frames_multichannel)
# #                 check_distortless_constraint(w, self.vs)
#             D[:, :, i_training_sample] = w
            
#         return D

#     def _initialize(self, X):
#         pass

#     def _choose_weights(self, x):
#         n_dictionary_atoms = self.weights_.shape[2]
#         min_ave_energy = np.inf
#         optimal_weight_index = None
#         for i_dictionary_atom in range(n_dictionary_atoms):
#             w_frequency = self.weights_[:, :, i_dictionary_atom]
#             energy = 0
#             n_fft_bins = w_frequency.shape[0]
#             for i_fft_bin in range(n_fft_bins):
#                 w = w_frequency[i_fft_bin]
#                 R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
#                 energy += np.real(w.transpose().conjugate().dot(R).dot(w))
#             ave_energy = energy / n_fft_bins
#             if min_ave_energy > ave_energy:
#                 min_ave_energy = ave_energy
#                 optimal_weight_index = i_dictionary_atom
#         optimal_weight = self.weights_[:, :, optimal_weight_index]
#         return optimal_weight, optimal_weight_index
    
#     def fit(self, training_data):
#         """
#         Parameters
#         ----------
#         X: shape = [n_samples, n_features]
#         """
#         D = self._compute_weights(training_data)
#         self.weights_ = D
#         return self

#     def choose_weights(self, x):
#         return self._choose_weights(x)    
import numpy as np
from dlbeamformer_utilities import compute_mvdr_tf_beamformers, check_distortless_constraint, compute_steering_vectors,\
compute_null_controlling_tf_beamformers, compute_null_controlling_minibatch_tf_beamformers,\
compute_null_controlling_tf_beamformers_2
from tqdm import tnrange, tqdm

class DictionaryLearningBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, diagonal_loading_param=1, bf_type="NC"):
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
        self.diagonal_loading_param = diagonal_loading_param
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
    
    def _compute_weights(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, batch_size=1, n_atoms_each_config=1):
        n_configurations = len(training_data)
        n_train_samples_each_config, n_fft_bins, n_mics, _ = training_data[0][0].shape
        n_sources = len(self.source_steering_vectors)
        D = np.zeros((n_sources, n_fft_bins, n_mics, n_configurations*n_atoms_each_config), dtype=complex)

        for i_source in range(n_sources):
            for i_configuration in tqdm(range(n_configurations), desc="Training configuration"):
                for i_atom in range(n_atoms_each_config):
                    batch_indices = np.random.choice(len(training_data[i_configuration][0]), batch_size, replace=True)
                    tf_sample_covariance_batch = training_data[i_configuration][0][batch_indices]
#                     print(training_data[i_configuration][0].shape, tf_sample_covariance_batch.shape, batch_indices)
                    null_angle_range = self._compute_null_angle_ranges(
                        training_data[i_configuration][1]["theta"], desired_null_width)
                    null_steering_vectors = compute_steering_vectors(
                        self.array_geometry, self.sampling_frequency,
                        self.stft_params["n_fft_bins"],
                        np.unique(null_angle_range), np.unique(training_data[i_configuration][1]["phi"])
                    )
                    null_steering_vectors = np.transpose(null_steering_vectors[:, :, 0, :], (0, 2, 1))
            
                    w = compute_null_controlling_tf_beamformers_2(
                            self.source_steering_vectors[i_source][:, 0, 0, :], null_steering_vectors, 
                            tf_sample_covariance_batch, 
                            null_constraint_threshold, 
                            eigenvalue_percentage_threshold=0.99, diagonal_loading_param=self.diagonal_loading_param)
                    D[i_source, :, :, i_configuration*n_atoms_each_config + i_atom] = w
            
        return D
    
    def _compute_null_angle_ranges(self, null_thetas, desired_null_width):
        theta_ranges = []
        for null_theta in null_thetas:
            theta_ranges.append(
                np.arange(null_theta - desired_null_width/2,
                          null_theta + desired_null_width/2, 0.1))
        return np.concatenate(theta_ranges)
            
#     def _initialize(self, X):
#         pass

    def _choose_weights(self, source_angle_index, x):
        weights_ = self.weights_[source_angle_index]
        n_fft_bins, n_mics, n_dictionary_atoms = weights_.shape
#         min_ave_energy = np.inf
#         optimal_weight_index = None
#         for i_dictionary_atom in range(n_dictionary_atoms):
#             w_frequency = weights_[:, :, i_dictionary_atom]
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
#         optimal_weight = weights_[:, :, optimal_weight_index]
        optimal_weights = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
        for i_fft_bin in tqdm(range(n_fft_bins), desc="FFT bin"):
            R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate()) + self.diagonal_loading_param*np.identity(n_mics)
            W = weights_[i_fft_bin]
            i_fft_optimal_weight_index = np.argmin(np.diagonal(np.abs(W.transpose().conjugate().dot(
                R).dot(W))))
            optimal_weights[i_fft_bin] = weights_[i_fft_bin, :, i_fft_optimal_weight_index]
        return optimal_weights
    
    def fit(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, batch_size=1, n_atoms_each_config=1):
        """
        Parameters
        ----------
        """
        D = self._compute_weights(training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold, batch_size, n_atoms_each_config)
        self.weights_ = D
        return self

    def choose_weights(self, source_angle_index, x):
        return self._choose_weights(source_angle_index, x)
    
    
class DLBatchBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, bf_type="NC"):
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
        print("Initialize DL Batch Beamformer")
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
    
    def _compute_weights(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_atoms_each_config=1):
        n_configurations = len(training_data)
        n_fft_bins, n_mics, _ = training_data[0][0][0].shape
        n_sources = len(self.source_steering_vectors)
        D = np.zeros((n_sources, n_fft_bins, n_mics, n_configurations*n_atoms_each_config), dtype=complex)
        for i_source in range(n_sources):
            for i_configuration in tqdm(range(n_configurations), desc="Configuration"):
                for i_atom in range(n_atoms_each_config):
                    train_data_indices = np.random.choice(len(training_data[i_configuration][0]), batch_size)
                    tf_sample_covariance_matrices = training_data[i_configuration][0][train_data_indices]
                    null_angle_range = self._compute_null_angle_ranges(
                        training_data[i_configuration][1]["theta"], desired_null_width)
                    null_steering_vectors = compute_steering_vectors(
                        self.array_geometry, self.sampling_frequency,
                        self.stft_params["n_fft_bins"],
                        np.unique(null_angle_range), np.unique(training_data[i_configuration][1]["phi"])
                    )
                    null_steering_vectors = np.transpose(null_steering_vectors[:, :, 0, :], (0, 2, 1))
                    w = compute_null_controlling_minibatch_tf_beamformers(
                            self.source_steering_vectors[i_source][:, 0, 0, :], null_steering_vectors, 
                            tf_sample_covariance_matrices, 
                            null_constraint_threshold, 
                            eigenvalue_percentage_threshold=0.99)
                    D[i_source, :, :, i_configuration*n_atoms_each_config + i_atom] = w
            
        return D
    
    def _compute_null_angle_ranges(self, null_thetas, desired_null_width):
        theta_ranges = []
        for null_theta in null_thetas:
            theta_ranges.append(
                np.arange(null_theta - desired_null_width/2,
                          null_theta + desired_null_width/2, 0.1))
        return np.concatenate(theta_ranges)
            
#     def _initialize(self, X):
#         pass

    def _choose_weights(self, source_angle_index, x):
        weights_ = self.weights_[source_angle_index]
        n_fft_bins, n_mics, n_dictionary_atoms = weights_.shape
#         min_ave_energy = np.inf
#         optimal_weight_index = None
#         for i_dictionary_atom in range(n_dictionary_atoms):
#             w_frequency = weights_[:, :, i_dictionary_atom]
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
#         optimal_weight = weights_[:, :, optimal_weight_index]
        optimal_weights = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
        for i_fft_bin in tqdm(range(n_fft_bins), desc="FFT bin"):
            R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
            W = weights_[i_fft_bin]
            i_fft_optimal_weight_index = np.argmin(np.diagonal(np.abs(W.transpose().conjugate().dot(
                R).dot(W))))
            optimal_weights[i_fft_bin] = weights_[i_fft_bin, :, i_fft_optimal_weight_index]
        return optimal_weights
    
    def fit(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_atoms_each_config=1):
        """
        Parameters
        ----------
        """
        D = self._compute_weights(training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold, 
            batch_size, n_atoms_each_config)
        self.weights_ = D
        return self

    def choose_weights(self, source_angle_index, x):
        return self._choose_weights(source_angle_index, x)
import numpy as np
from dlbeamformer_utilities import compute_mvdr_tf_beamformers, check_distortless_constraint
from tqdm import tnrange, tqdm_notebook

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
        for i_training_sample in tqdm_notebook(range(n_training_samples), desc="Training sample"):
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
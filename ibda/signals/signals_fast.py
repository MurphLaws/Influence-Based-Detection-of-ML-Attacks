import operator
from typing import Callable

import numpy as np
import pandas as pd


class InfluenceErrorSignals:
    def __init__(self,
                 train_test_inf_mat: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 compute_test_influence: bool):

        self.train_test_inf_mat = train_test_inf_mat.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self._unq_labels = np.unique(self.y_train)
        self.acni_opt_labels = None
        self.cpi_opt_labels = None

        if compute_test_influence:
            self.train_test_inf_mat = self.train_test_inf_mat.T
            self.y_train, self.y_test = self.y_test, self.y_train

        self.__signals = {
            'CPI': self.cpi,
            'ACNI': self.acni,
            'ACNI^*': self.acni_opt,
            'CPI^*': self.cpi_opt,
            'ACI': self.aci,
            'MNI': self.mani,
            'MPI': self.mpi,
            'MTI': self.mti,
            'MNIC': self.mnic,
        }

    def _conditional_influences(self) -> np.ndarray:
        train_test_inf_mat_local = self.train_test_inf_mat.copy()
        conditional_inf_values = np.zeros(train_test_inf_mat_local.shape)
        for l in self._unq_labels:
            l_train_ids = np.where(self.y_train == l)[0]
            l_test_ids = np.where(self.y_test == l)[0]
            l_train_ids_broadcasted = np.broadcast_to(l_train_ids[:, np.newaxis], (len(l_train_ids), len(l_test_ids)))
            inf_of_l_samples = train_test_inf_mat_local[l_train_ids, :]
            conditional_inf_values[l_train_ids_broadcasted, l_test_ids] = inf_of_l_samples[:, l_test_ids]
        return conditional_inf_values

    def compute_signals(self, verbose=True) -> pd.DataFrame:
        signal_vals = {}
        for sig_name, sig_fn in self.__signals.items():
            if verbose:
                print(sig_name)
            signal_vals[sig_name] = sig_fn()
        return pd.DataFrame.from_dict(signal_vals)

    #####################
    # Conditional Signals

    # Conditional Positive Influence
    def cpi(self) -> np.ndarray:
        cond_influences = self._conditional_influences()
        cond_influences[cond_influences < 0] = 0
        return cond_influences.sum(axis=1)

    # Absolute Conditional Negative Influence
    def acni(self) -> np.ndarray:
        cond_influences = self._conditional_influences()
        cond_influences[cond_influences > 0] = 0
        return np.abs(cond_influences.sum(axis=1))

    def acnic(self) -> np.ndarray:
        cond_influences = self._conditional_influences()
        cond_influences[cond_influences > 0] = 0
        return np.where(cond_influences != 0)[0]

    # Aboslute Conditional Influence
    def aci(self) -> np.ndarray:
        return self.cpi() + self.acni()

    # Absolute Conditional Negative Influence Optimal
    def acni_opt(self) -> np.ndarray:
        train_test_inf_mat_tmp = self.train_test_inf_mat.copy()
        nil_values = None
        for l in self._unq_labels:
            l_test_ids = np.where(self.y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
                ] = 0
            nil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if nil_values is None:
                nil_values = nil_values_tmp.copy()
            else:
                nil_values = np.vstack((nil_values, nil_values_tmp))
            self.acni_opt_labels = nil_values.argmin(axis=0)
        return np.abs(nil_values.min(axis=0))

    # Absolute Conditional Positive Influence Optimal
    def cpi_opt(self) -> np.ndarray:
        train_test_inf_mat_tmp = self.train_test_inf_mat.copy()
        cpi_values = None
        for l in self._unq_labels:
            l_test_ids = np.where(self.y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples < 0
                ] = 0
            cpi_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if cpi_values is None:
                cpi_values = cpi_values_tmp.copy()
            else:
                cpi_values = np.vstack((cpi_values, cpi_values_tmp))
            self.cpi_opt_labels = cpi_values.argmax(axis=0)
        return cpi_values.max(axis=0)

    ##################
    # Marginal Signals

    # Marginal Positive Influence
    def mpi(self):
        train_test_inf_mat_tmp = self.train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp < 0] = 0
        return train_test_inf_mat_tmp.sum(axis=1)

    # Absolute Marginal Negative Influence
    def mani(self):
        train_test_inf_mat_tmp = self.train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp > 0] = 0
        return np.abs(train_test_inf_mat_tmp.sum(axis=1))

    # Marginal Total Influence
    def mti(self):
        return self.train_test_inf_mat.sum(axis=1)

    # Marginal Negative Influence Count
    def mnic(self):
        train_test_inf_mat_tmp = self.train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp > 0] = 0
        train_test_inf_mat_tmp[train_test_inf_mat_tmp < 0] = 1
        return train_test_inf_mat_tmp.sum(axis=1)



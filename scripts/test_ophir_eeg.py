import numpy as np
from mne.decoding import CSP
from mne_features.feature_extraction import FeatureExtractor
from mne.channels import make_standard_montage
import mne
from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import scipy.io

def laplace(arr):
    """Laplace filtered signal as features.

    Parameters
    ----------
    arr : ndarray, shape (n_channels, n_times)

    Returns
    -------
    output : (n_channels * n_times,)
    """
    c4 = 4 * arr[8] - (arr[2] + arr[7] + arr[14] + arr[9])
    c3 = 4 * arr[4] - (arr[0] + arr[3] + arr[10] + arr[5])

    return np.array([np.mean(c3), np.mean(c4)])

def cross_validation(clf, trials, labels):

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    # scoring = ('r2', 'neg_mean_squared_error')

    cv_results = cross_validate(clf, trials, labels, cv=kf, scoring='accuracy',
                                return_train_score=False)
    print(cv_results['test_score'].mean())


def csp_lda(trials, labels):
    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline
    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # fit transformer and classifier to data
    clf.fit(trials, labels)
    print("csp_lda")
    # cross_validation(clf, trials, labels)

    X_r2 = clf.fit(trials, labels).transform(trials)
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [1, 2, 3], ["left", "idle", "right"]):
        plt.scatter(
            X_r2[labels == i, 0], X_r2[labels == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA Separation")

    plt.show()

def random_forest(trials, labels):
    selected_funcs = ['max_cross_corr','time_corr', 'pow_freq_bands']

    clf = make_pipeline(FeatureExtractor(sfreq=500,
                                              selected_funcs=selected_funcs),
                        RandomForestClassifier())
    print("random_forest")
    cross_validation(clf, trials, labels)

def svm(trials, labels):
    selected_funcs = ['time_corr',"max_cross_corr", "pow_freq_bands"]

    # selected_funcs = ['pow_freq_bands', ('laplace', laplace)]
    labels = labels - 1
    clf = make_pipeline(FeatureExtractor(sfreq=500,
                                              selected_funcs=selected_funcs),
                             StandardScaler(),
                             SVC(gamma='auto'))
    # clf.fit(trials, labels)
    print("svm")
    cross_validation(clf, trials, labels)

def plot_electrodes(amplitude):
    plt.figure()
    plt.plot(range(len(amplitude)), amplitude)
    plt.show()

def main():
    data_folder = "C:\\Users\\Elad\\bci4als-master_old\\recordings\\ophir_eeg"
    trials = scipy.io.loadmat(f'{data_folder}/EEG.mat')['EEG']
    # make it of shape n_epochs, n_channels, n_times 16,512,85
    trials = np.transpose(trials, (2, 0, 1))
    labels = scipy.io.loadmat(f'{data_folder}/trainingVec.mat')['trainingVec']
    labels = labels.reshape(-1)


    ch_names = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Fz']
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types=ch_types)
    epochs = mne.EpochsArray(trials, info)

    # set montage
    montage = make_standard_montage('standard_1020')
    epochs.set_montage(montage)

    #plot_electrodes(trials[12][4])

    # Apply band-pass filter
    epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    trials = epochs.get_data()
    plot_electrodes(trials[12][4])

    # trials = trials[:, [1,2,3], :]
    # selected_funcs = ['pow_freq_bands', 'samp_entropy', 'spect_entropy', 'spect_slope']
    # fe = FeatureExtractor(sfreq=500,
    #                  selected_funcs=selected_funcs)
    # x = fe.fit_transform(trials, labels)

    # csp_lda(trials, labels)
    # svm(trials, labels)
    random_forest(trials, labels)

if __name__ == "__main__":
    main()

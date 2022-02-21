import os
import pickle
import numpy as np
from numpy import genfromtxt

from tkinter.filedialog import askdirectory

def main():
    init_dir = os.path.join(os.path.split(os.path.abspath(''))[0], 'recordings')
    trials_folder = askdirectory(initialdir=init_dir)
    labels = []
    trials = []
    for subdir, dirs, files in os.walk(trials_folder):
        label_file = next((x for x in files if x=="labels.csv"), None)
        trials_file = next((x for x in files if x=="trials.pickle"), None)
        if label_file and trials_file:
            label_file = os.path.join(subdir, label_file )
            trials_file = os.path.join(subdir, trials_file)
            labels.append(genfromtxt(label_file, delimiter=','))
            trials += pickle.load(open(trials_file, 'rb'))

    n_samples = min([t.shape[0] for t in trials])

    np_array = list(map(lambda x: x.to_numpy(), trials))
    np_array = list(map(lambda x: x[:n_samples, :], np_array))
    np_array = np.transpose(np_array, (0,2,1))

    all_labels = np.concatenate([label for label in labels])

    pickle.dump(all_labels, open(f'{trials_folder}/labels.pickle', 'wb'))
    pickle.dump(np_array, open(f'{trials_folder}/trials.pickle', 'wb'))

if __name__ == "__main__":
    main()

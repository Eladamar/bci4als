import pickle

from bci4als.ml_model import MLModel
from bci4als.experiments.online import OnlineExperiment
from bci4als.eeg import EEG


def run_experiment(model_path: str):

    model = pickle.load(open(model_path, 'rb'))

    SYNTHETIC_BOARD = -1
    CYTON_DAISY = 2
    eeg = EEG(board_id=CYTON_DAISY)

    exp = OnlineExperiment(eeg=eeg, model=model, num_trials=10, buffer_time=3, threshold=1, skip_after=4,
                           co_learning=True, debug=False, classes=(0,1,2))

    exp.run(use_eeg=True, full_screen=False)

    # pickle.dump(model, open('./model.pickle', 'wb'))


if __name__ == '__main__':

    model_path = 'C:\\Users\\Elad\\bci4als\\recordings\\shiri10.3nd\\1f\\model.pickle'
    # model_path = None  # use if synthetic
    run_experiment(model_path=model_path)

# PAY ATTENTION!
# If synthetic - model Path should be none
# otherwise choose a model path

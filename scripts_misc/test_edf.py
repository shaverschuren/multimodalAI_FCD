import pyedflib

def test_edf(filename):
    reader = pyedflib.EdfReader(filename)
    n = reader.signals_in_file
    signal_labels = reader.getSignalLabels()
    print(f"Number of signals: {n}")
    print("Signal labels:")
    for i in range(n):
        print(f"  {i}: {signal_labels[i]}")
    reader.close()

if __name__ == "__main__":
    test_edf(r"L:\her_knf_golf\Wetenschap\newtransport\Sjors\data\raw\eeg\trc2edf\RESP0372\EEG_135054.edf")
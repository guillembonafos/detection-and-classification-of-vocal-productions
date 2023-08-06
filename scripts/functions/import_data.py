import librosa
import pandas as pd


def import_raw_data(metadata, sr):
    """
    Import the raw data, wave files.

    Parameters
    ----------
    metadata : pd.Dataframe(n,4)
        DataFrame with metadata of the audio files. Must contain at least 2 columns:
        label (label of the recordings), fd (path to the recordings).
    sr : int
        Sampling rate of the recordings, in Hertz.

    Returns
    -------
    df : pd.DataFrame(n, 2)
        DataFrame with the wav files and their label.
    """
    fichiers = []
    for index, row in metadata.dropna().iterrows():
        label = row['label']
        y, sr = librosa.load(row['fd'], sr=sr)
        fichiers.append([y, label, index])
        print(index)
    return pd.DataFrame(fichiers, columns=['audio', 'label', 'index'])

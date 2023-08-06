import numpy as np
import pandas as pd
from functools import partial
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow_hub as hub
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def preparation_data_set(meta, resample=True, batch_size=32, transfer_learning=True):
    """
    Take the metadata and prepare the tensorflow dataset, pipeline to load the
    data, for the learning of the model.

    Parameters
    ----------
    meta : pd.DataFrame
        DataFrame with the useful informations: path to the recordinfs, the label
        of the detection problem (voc or noise), the label of the classification
        problem (which type of voc, or a random if noise), the weighting (1/2 for
        the binary problem, 1/n_class for the multi problem).
    resample : bool
        If True, the dataset is an infinite dataset in which the distribution of
        each label is uniform. We compute the number of batch to see each frame
        of the most important class once (to fix when to stop the sample in this
        infinite ds during the learning).
        If False, the dataset outputs each frame once, without resampling and
        uniform distribution.
    batch_size : int
        Size of batch.
    transfer_learning : bool
        If True, take the front-end of the pretrained YamNet model to build a
        latent representation space of the signal.
        If False, outputs the raw data.

    Returns
    -------
    ds : tf.dataset
        Iterator, each element is one second of audio, for which we compute a
        representation from YamNet front-end (1024,) or the raw audio wav file
        (16000), a dictionnary with the labels for the binary and multi problems,
        a dictionnary with the weights for the binary and multi problems.
    """
    if transfer_learning is True:
        # Link to yamnet
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
    else:
        pass

    def get_waveform(file_path, targets, weights):
        """
        Load the audio file. Values normalized between -1 and 1. Resampled to
        16 kHz. Transformed into mono.

        Parameters
        ----------
        file_path : str
            Path to the audio recordings.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binar problems, respectively output_multi
            and output_binaire.

        Returns
        -------
        audio : tf.Tensor
            Tensor ith the wav file, sampled to 16000 in mono.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.
        """
        audio_binary = tf.io.read_file(file_path)
        audio, sr_in = tf.audio.decode_wav(audio_binary)
        audio = tfio.audio.resample(input=audio, rate_in=tf.cast(sr_in, tf.int64), rate_out=16000)
        audio = tf.reduce_mean(audio, axis=1)

        return audio, targets, weights

    def windowing(waveform, targets, weights):
        """
        Rolling window of 1 second with an 80% overlap.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.

        Returns
        -------
        audio : tf.Tensor
            The audio split in several frame with the overlapping window.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.
        """
        frames = tf.signal.frame(waveform, frame_length=16000, frame_step=3200,
                                 pad_end=True, pad_value=0, axis=-1)
        n_frames = tf.shape(frames)[0]
        return frames, \
               {'output_multi': tf.repeat(targets['output_multi'], n_frames),
                'output_binaire': tf.repeat(targets['output_binaire'], n_frames)}, \
               {'output_multi': tf.repeat(weights['output_multi'], n_frames),
                'output_binaire': tf.repeat(weights['output_binaire'], n_frames)}

    def coding(waveform, targets, weights, n_classes):
        """
        Encode the labels of the classification problem into indicator (one hot
        encoding).

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.
        n_classes : int
            NUmber of labels of the classification problem. We compute it outside
            of the function through the partial function.

        Returns
        -------
        audio : tf.Tensor
            The audio split in several frame with the overlapping window.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire. Both are one-hot-encoding.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.
        """
        return waveform, {'output_multi': tf.one_hot(targets['output_multi'], depth=n_classes, dtype=tf.float32),
                          'output_binaire': targets['output_binaire']}, weights

    def representation(waveform, targets, weights):
        """
        Compute representations of the audios using the YamNet model pre-trained
        on the Audioset corpus.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.

        Returns
        -------
        embeddings : tf.Tensor
            Embedding of the frame using the transfered network. The front-end
            outputs two embeddings for a 1 second frame. We take the mean.
        targets : dict
            Code of the labels for the multi and binary problems, respectively
            output_multi and output_binaire.
        weights : dict
            Weights for the multi and binary problems, respectively output_multi
            and output_binaire.
        """
        scores, embeddings, spectrogram = yamnet_model(waveform)
        embeddings = tf.reduce_mean(embeddings, axis=0)
        return embeddings, targets, weights

    def equilibrage_classes(meta, batch_size, transfer_learning):
        """
        Create a dataset per label, with the steps of the pipeline, and joint them
        sampling in such that the distribution of the training set is uniform.

        Parameters
        ----------
        meta : pd.DatFrame
            Path to the records.
        batch_size : int
            Size of batches that we want.
        transfer_learning : bool
            Do we use transfer-learning.

        Returns
        -------
        ds : tf.Dataset
            Dataset outputs a training set with an uniform distribution per class.
        """
        # Names of the labels for the problem
        classes = np.unique(meta["label"])

        ds_classes = {}
        n_classes = len(classes)
        for classe in classes:
            ds_classes['ds_' + str(classe)] = tf.data.Dataset.from_tensor_slices((meta[(meta["label"] == classe) & (meta["label_binaire"] == "voc")]["fd"],
                                                                                 {'output_multi': meta[(meta["label"] == classe) & (meta["label_binaire"] == "voc")]["classe_code_multi"],
                                                                                  'output_binaire': meta[(meta["label"] == classe) & (meta["label_binaire"] == "voc")]["classe_code_binaire"]},
                                                                                 {'output_multi': meta[(meta["label"] == classe) & (meta["label_binaire"] == "voc")]["weight_multi"],
                                                                                  'output_binaire': meta[(meta["label"] == classe) & (meta["label_binaire"] == "voc")]["weight_binaire"]}))\
                .map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(partial(coding, n_classes=n_classes), num_parallel_calls=tf.data.AUTOTUNE).cache()
        ds_classes['ds_noise'] = tf.data.Dataset.from_tensor_slices((meta[meta["label_binaire"] == "noise"]["fd"],
                                                                     {'output_multi': meta[meta["label_binaire"] == "noise"]["classe_code_multi"],
                                                                      'output_binaire': meta[meta["label_binaire"] == "noise"]["classe_code_binaire"]},
                                                                     {'output_multi': meta[meta["label_binaire"] == "noise"]["weight_multi"],
                                                                      'output_binaire': meta[meta["label_binaire"] == "noise"]["weight_binaire"]}))\
            .map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(partial(coding, n_classes=n_classes), num_parallel_calls=tf.data.AUTOTUNE).cache()

        # Heuristics to compute the number of epoch : the number of batch to see
        # each frame of the most representative class.
        # To save time, we do not compute the number of frame of each class, we
        # know that the most representative class is the noise class
        l_noise = ds_classes["ds_noise"].reduce(0, lambda x, _: x + 1).numpy()
        resampled_steps_per_epoch = np.ceil(n_classes * l_noise / batch_size)

        if transfer_learning is True:
            for classe in classes:
                ds_classes["ds_" + str(classe)] = ds_classes["ds_" + str(classe)].map(representation, num_parallel_calls=tf.data.AUTOTUNE).cache().repeat()
            ds_classes["ds_noise"] = ds_classes["ds_noise"].map(representation, num_parallel_calls=tf.data.AUTOTUNE).cache().repeat()
        else:
            pass

        # We compute the weighting for the classification problem. Equiprobability
        # between each label but also between all the vocs and the noise.
        weight_multi = (1/n_classes)/2
        weights = np.repeat(weight_multi, n_classes).tolist()
        # We add wighting for the noise class, 1/2 (as much noise as voc). The
        # noise dataset is the last of the dictionnary, we append.
        weights.append(0.5)

        # We create the final dataset sampling in the labels datasets
        ds = tf.data.experimental.sample_from_datasets(list(ds_classes.values()), weights=weights)

        return ds, resampled_steps_per_epoch

    if resample is True:
        ds, resampled_steps_per_epoch = equilibrage_classes(meta, batch_size, transfer_learning)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), resampled_steps_per_epoch

    else:
        classes = np.unique(meta["label"])
        n_classes = len(classes)
        ds = tf.data.Dataset.from_tensor_slices((meta["fd"],
                                                 {'output_multi': meta["classe_code_multi"], 'output_binaire': meta["classe_code_binaire"]},
                                                 {'output_multi': meta["weight_multi"], 'output_binaire': meta["weight_binaire"]}))
        if transfer_learning is True:
            ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(partial(coding, n_classes=n_classes), num_parallel_calls=tf.data.AUTOTUNE).map(representation, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(partial(coding, n_classes=n_classes), num_parallel_calls=tf.data.AUTOTUNE)

        return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preparation_longform_papio(meta, batch_size=32, transfer_learning=True):
    """
    Outputs the pipeline to predict the long form audio recordings from their
    metadata. Split them into 1 second frame with an 80% overlapping window.
    Outputs the one second wwav, the day and the hour.

    Parameters
    ----------
    meta : pd.DataFrame
        DataFrame with the useful information: path to the recordings, the day
        and the hour.
    batch_size : int
        Batch size wanted for the prediction.
    transfer_learning : bool
        If True, use transfer-learning with the fron-end of the YamNet model.
        If False, raw data.

    Returns
    -------
    ds : tf.Dataset
        Iterator, each element is one second of audio, for which we compute a
        representation from YamNet front-end (1024,) or the raw audio wav file
        (16000), the day and the hour.
    """
    if transfer_learning is True:
        # Lin to YamNet
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
    else:
        pass

    def get_waveform(file_path, day, hour):
        """
        Load the audio file. Values normalized between -1 and 1. Resampled to
        16 kHz. Transformed into mono.

        Parameters
        ----------
        file_path : str
            Path to the audio recordings.
        day : str
            Day of the recordings.
        hour : str
            Hour of the recordings.

        Returns
        -------
        audio : tf.Tensor
            Tensor with the wave file, sampled to 16 kHz.
        day : str
            Jour de l'enregistrement
        hour : str
            Heure du début et de la fin de l'enregistrement.
        """
        audio_binary = tf.io.read_file(file_path)
        audio, sr_in = tf.audio.decode_wav(audio_binary)
        audio = tfio.audio.resample(input=audio, rate_in=tf.cast(sr_in, tf.int64), rate_out=16000)
        audio = tf.reduce_mean(audio, axis=1)

        return audio, day, hour

    def windowing(waveform, day, hour):
        """
        Rolling window of 1 second with an 80% overlap.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        day : str
            Day of the recordings.
        hour : str
            Hour of the recordings.

        Returns
        -------
        audio : tf.Tensor
            Audio, split with an 80% overlapping 1 second window.
        day : str
            Jour de l'enregistrement
        hour : str
            Heure du début et de la fin de l'enregistrement.
        """
        frames = tf.signal.frame(waveform, frame_length=16000, frame_step=3200,
                                 pad_end=True, pad_value=0, axis=-1)

        n_frames = tf.shape(frames)[0]
        return frames, tf.repeat(day, n_frames), tf.repeat(hour, n_frames)

    def representation(waveform, day, hour):
        """
        Compute representations of the audios using the YamNet model pre-trained
        on the Audioset corpus.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        day : str
            Day of the recordings.
        hour : str
            Hour of the recordings.

        Returns
        -------
        embeddings : tf.Tensor
            Embedding of the frame using the transfered network. The front-end
            outputs two embeddings for a 1 second frame. We take the mean.
        day : str
            Day of the recordings.
        hour : str
            Hour of the recordings.
        """
        scores, embeddings, spectrogram = yamnet_model(waveform)
        embeddings = tf.reduce_mean(embeddings, axis=0)
        return embeddings, day, hour

    ds = tf.data.Dataset.from_tensor_slices((meta["fd"], meta["day"], meta["hour"]))

    if transfer_learning is True:
        ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(representation, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preparation_longform_baby(meta, batch_size=32, transfer_learning=True):
    if transfer_learning is True:
        # Link to YamNet
        yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        yamnet_model = hub.load(yamnet_model_handle)
    else:
        pass

    def get_waveform(file_path, date, child):
        """
        Load the audio file. Values normalized between -1 and 1. Resampled to
        16 kHz. Transformed into mono.

        Parameters
        ----------
        file_path : str
            Path to the audio file.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.

        Returns
        -------
        audio : tf.Tensor
            Tensor of the wav file, sampled to 16 kHz in mono.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.
        """
        audio_binary = tf.io.read_file(file_path)
        audio, sr_in = tf.audio.decode_wav(audio_binary)
        audio = tfio.audio.resample(input=audio, rate_in=tf.cast(sr_in, tf.int64), rate_out=16000)
        audio = tf.reduce_mean(audio, axis=1)

        return audio, date, child

    def windowing(waveform, date, child):
        """
        Rolling window of 1 second with an 80% overlap.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.

        Returns
        -------
        audio : tf.Tensor
            Audio, split with an 80% overlapping 1 second window.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.
        """
        frames = tf.signal.frame(waveform, frame_length=16000, frame_step=3200,
                                 pad_end=True, pad_value=0, axis=-1)

        n_frames = tf.shape(frames)[0]
        return frames, tf.repeat(date, n_frames), tf.repeat(child, n_frames)

    def representation(waveform, date, child):
        """
        Compute representations of the audios using the YamNet model pre-trained
        on the Audioset corpus.

        Parameters
        ----------
        waveform : tf.Tensor
            Wav file.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.

        Returns
        -------
        embeddings : tf.Tensor
            Embedding of the frame using the transfered network. The front-end
            outputs two embeddings for a 1 second frame. We take the mean.
        date : str
            Date of the recordings.
        child : str
            Identificator of the child recorded.
        """
        scores, embeddings, spectrogram = yamnet_model(waveform)
        embeddings = tf.reduce_mean(embeddings, axis=0)

        return embeddings, date, child

    ds = tf.data.Dataset.from_tensor_slices((meta["fd"], meta["date"].astype("string"), meta["child"]))

    if transfer_learning is True:
        ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch().map(representation, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(get_waveform, num_parallel_calls=tf.data.AUTOTUNE).map(windowing, num_parallel_calls=tf.data.AUTOTUNE).unbatch()

    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

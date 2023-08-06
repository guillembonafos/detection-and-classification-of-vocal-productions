#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:21:11 2019

@author: guillem
"""

import pandas as pd
import os
import numpy as np
import sklearn as sk
import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def meta_files(fd):
    """
    Metadata of recordings: path to each recordings and its class.

    Parameters
    ----------
    fd : basestring
        Path to the recordings.

    Returns
    -------
    metadonnees : pd.DataFrame(n, 4)
        DataFrame
    """
    chemin = os.path.join(fd, 'data/vocalizations/')

    # DataFrame to summarize the results
    metadonnees = pd.DataFrame(columns=['fd', 'label'])

    # Set of the name files of the recordings
    metadonnees['fd'] = [f for f in os.listdir(chemin)]

    # we take the absolute path and we add the class of the recordings
    for index, row in metadonnees.iterrows():
        metadonnees.loc[index, 'fd'] = os.path.join(chemin, str(row['fd']))
        if 'bark' in row['fd']:
            metadonnees.loc[index, 'label'] = 'bark'
        elif 'grunt' in row['fd']:
            metadonnees.loc[index, 'label'] = 'grunt'
        elif 'copulation' in row['fd']:
            metadonnees.loc[index, 'label'] = 'copulation'
        elif 'scream' in row['fd']:
            metadonnees.loc[index, 'label'] = 'scream'
        elif 'wahoo' in row['fd']:
            metadonnees.loc[index, 'label'] = 'wahoo'
        elif 'yak' in row['fd']:
            metadonnees.loc[index, 'label'] = 'yak'
        elif 'bruit' in row['fd']:
            metadonnees.loc[index, 'label'] = 'noise'

    return metadonnees.dropna().reset_index(drop=True)


def metadata_augmentation(fd):
    """
    Metadata of the recordings of the data augmentation. The data augmentation
    must have been done and the new recordings split in three folders, one per
    modality (noise, tonality, speed).

    Parameters
    ----------
    fd : string
        Path to the recordings.

    Returns
    -------
    metadonnees_aug : pd.DataFrame(n, 3)
        DataFrame
    """
    metadonnees_aug = pd.DataFrame(columns=['chemin_relatif', 'modalite', 'label'])

    modalite = ['noise', 'tonality', 'speed']

    for mod in modalite:
        result = pd.DataFrame(columns=['chemin_relatif', 'modalite', 'label'])
        fichiers = [f for f in os.listdir(os.path.join(fd, mod))]
        result['chemin_relatif'] = fichiers
        result['modalite'] = mod
        for f, i in zip(result['chemin_relatif'], range(0, len(fichiers))):
            if 'bark' in f:
                result.label[i] = 'bark'
            elif 'grunt' in f:
                result.label[i] = 'grunt'
            elif 'copulation' in f:
                result.label[i] = 'copulation'
            elif 'scream' in f:
                result.label[i] = 'scream'
            elif 'wahoo' in f:
                result.label[i] = 'wahoo'
            elif 'yak' in f:
                result.label[i] = 'yak'
            elif 'bruit' in f:
                result.label[i] = 'noise'

        metadonnees_aug = pd.concat([metadonnees_aug, result])
        metadonnees_aug = metadonnees_aug.reset_index(drop=True)

    metadonnees_aug['fd'] = fd + metadonnees_aug['modalite'] + '/' + metadonnees_aug['chemin_relatif']

    return metadonnees_aug.reset_index(drop=True)



def meta_papio(fd, data_augmentation=True, two_loss=True, weighting_sampling=False):
    """
    Preparation of the dataframe which summarize the metadata fo the learning.

    Parameters
    ----------
    fd : basestring
        Path to the recordings.
    data_augmentation : bool
        True if we did data augmentation. In that case, the data augmented records
        are in a folder fd/data/data_augmentation/, one for train, one for validation,
        one for test set.
    two_loss : bool
        True if we want to train a model with two outputs, the prediction of the
        vocalization and the prediction of the class of vocalization.
        If False, just information if it is a vocalization or not.
    weighting_sampling : bool
        If True, add a variable which tells the proportion of the class the observation
        belongs.

    Returns
    -------
    meta_train : pd.DataFrame
        Df with metadata of the training set.
    meta_val : pd.DataFrame
        Df with metadata of the validation set.
    meta_test : pd.DataFrame
        Df with metadata of the testing set.
    """
    # All the metadata
    meta = meta_files(fd)

    # Split train, val, test
    train, test = train_test_split(meta, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.25, random_state=1)

    if data_augmentation is True:
        # Metadata of the data augmentation
        aug_train = metadata_augmentation(os.path.join(fd, 'data/data_augmentation/train/'))
        aug_val = metadata_augmentation(os.path.join(fd, 'data/data_augmentation/val/'))
        # Append everything
        train = train.append(aug_train[['label', 'fd']])
        val = val.append(aug_val[['label', 'fd']])
    else:
        pass

    # Variable coding the classes for the classication problem
    le = sk.preprocessing.LabelEncoder()
    if two_loss is True:
        classes = np.unique(train[train["label"] != "noise"]["label"])
    else:
        classes = np.unique(train["label"])

    def creation_labels_id(data, two_loss, weighting_sampling):
        # Creation of the label variable for the binary problem, voc vs non-voc
        if two_loss is True:
            data = data.assign(label_binaire=np.where(data["label"] != "noise", "voc", data["label"]))
            if weighting_sampling is True:
                data = data.assign(weight_binaire=np.where(data["label_binaire"] == "noise", 0.6, 3.03),
                                   weight_multi=np.where(data["label"] == "noise", 0, 0.17))
                for classe, weight in zip(classes, [3.65, 4.96, 0.35, 0.46, 3.91, 4.31]):
                    data.loc[(data["label_binaire"] != "noise") & (data["label"] == classe), "weight_multi"] = weight
            else:
                data = data.assign(weight_binaire=0.5,
                                   weight_multi=np.where(data["label"] == "noise", 0, 0.17))
            # Modification of the label variable for the classification: when it is
            # noise, we assign to it with uniform probability the label of another
            # class (the binary variable being unchanged, we keep the information)
            data = data.assign(label=np.where(data["label"] == "noise", np.random.choice(classes, size=len(data)), data["label"]))
            # Creation of variables coding the class
            data = data.assign(classe_code_multi=le.fit_transform(data["label"]), classe_code_binaire=le.fit_transform(data["label_binaire"]))
        else:
            data = data.assign(classe_code=le.fit_transform(data["label"]))
            if weighting_sampling is True:
                for classe, weight in zip(classes, [18.98, 0.17, 25.75, 1.82, 2.38, 20.31, 22.38]):
                    data.loc[data["label"] == classe, "weight"] = weight
            else:
                data = data.assign(weight=1)

        return data

    metas = [creation_labels_id(df, two_loss, weighting_sampling) for df in [train, val, test]]

    return metas[0], metas[1], metas[2]


def meta_baby(fd, meta, data_augmentation=True):
    """
    Take the useful metadata for the learning. The path should go where the recordings
    are. In this folder, we need to find the file in which whe have information
    about the labeled recordings, a folder in which there are the noise recordings,
    a folder of the data-augmentation. In the latter, there are two other folders,
    train and val. In each, there are three other folders, one by modality of data
    augmentation (noise, tonality, speed).

    Parameters
    ----------
    fd : str
        Path to the recordings. We need to find two folders, one for the noise
        recordings, one for the data_augmentation.
    meta : pd.DataFrame
        DataFrame of the metadata of the recordings, with at least two columns:
        fd (absolute path to the file of the recordings) and label (class of the
        recordings).
    data_augmentation : bool
        If True, data augmentation has been done previously and the directory is
        organized as described previously. The metadata of these recordings are
        taken.

    Returns
    -------
    meta_train : pd.DataFrame
        Df with metadata of the training set.
    meta_val : pd.DataFrame
        Df with metadata of the validation set.
    meta_test : pd.DataFrame
        Df with metadata of the testing set.
    """
    # Partition train/test/val
    meta_train, meta_test = train_test_split(meta, test_size=0.2, random_state=1)
    meta_train, meta_val = train_test_split(meta_train, test_size=0.25, random_state=1)

    if data_augmentation is True:
        # df for the metadata of the data augmentation
        modalite = ['noise', 'tonality', 'speed']
        metadonnees_aug = pd.DataFrame(columns=["fd", "modalite", "label"])
        for mod in modalite:
            fichiers = [f for f in os.listdir(os.path.join(fd, "data_augmentation/train", mod))]
            tmp = pd.DataFrame(columns=["fd", "modalite", "label"])
            tmp["fd"] = fichiers
            tmp["modalite"] = mod
            metadonnees_aug = pd.concat([metadonnees_aug, tmp])

        # absolute path
        metadonnees_aug["fd"] = fd + "/data_augmentation/train/" + metadonnees_aug["modalite"] + "/" + metadonnees_aug["fd"]

        # correspondance of the labels
        metadonnees_aug.reset_index(inplace=True)
        for index, row in meta_train.iterrows():
            metadonnees_aug.loc[metadonnees_aug[metadonnees_aug["fd"].str.contains(os.path.basename(meta_train.loc[index, "fd"])[:-4])].index, "label"] = row["label"]
    else:
        pass

    # add the noise recordings
    meta_bruit = pd.DataFrame({"fd": [os.path.join(fd, "noise", f) for f in os.listdir(os.path.join(fd, "noise"))], "label": "noise"})
    train_bruit, test_bruit = train_test_split(meta_bruit, test_size=0.2, random_state=1)
    train_bruit, val_bruit = train_test_split(train_bruit, test_size=0.25, random_state=1)

    # merge everything
    if data_augmentation is True:
        meta_train = pd.concat([meta_train, metadonnees_aug, train_bruit])
    else:
        meta_train = pd.concat([meta_train, train_bruit])
    meta_val = pd.concat([meta_val, val_bruit])
    meta_test = pd.concat([meta_test, test_bruit])

    # creation of the binary class
    meta_train, meta_test, meta_val = [df.assign(label_binaire=np.where(df["label"] != "noise", "voc", df["label"])) for df in [meta_train, meta_test, meta_val]]

    # weighting
    classes = meta_train[meta_train["label"] != "noise"]["label"].value_counts().index.values
    meta_train, meta_test, meta_val = [df.assign(weight_binaire=0.5, weight_multi=np.where(df["label"] == "noise", 0, 1/len(classes))) for df in [meta_train, meta_test, meta_val]]

    # attribution of a random label for the classification problem when the label
    # for the detection problem is noise
    meta_train, meta_test, meta_val = [df.assign(label=np.where(df["label"] == "noise", np.random.choice(classes, size=len(df)), df["label"])) for df in [meta_train, meta_test, meta_val]]

    # Coding the labels
    le = sk.preprocessing.LabelEncoder()
    [df.dropna(subset=["label"], inplace=True) for df in [meta_train, meta_test, meta_val]]
    meta_train, meta_test, meta_val = [df.assign(classe_code_multi=le.fit_transform(df["label"]), classe_code_binaire=le.fit_transform(df["label_binaire"])) for df in [meta_train, meta_test, meta_val]]

    return meta_train.reset_index(), meta_test.reset_index(), meta_val.reset_index()


def metadata_longform_papio(path='/media/guillhem/BackUParnaud/LOG_DATA', length=False):
    """
    Metadata of the long form audio recordings. Create a dataframe with as many
    rows as recordings, with three variables for each: the path, the hour of the
    recordings and the day.

    Parameters
    ----------
    path : string
        Path to go to the file where the recordings are.
    length : bool
        If True, load all the files to compute their length and the number of
        frames.

    Returns
    -------
    metadata : pd.DataFrame(n, 3)
        DataFrame
    """
    # List of the recordings
    paths = [os.path.join(path, f) for f in os.listdir(path)]

    # Date of the recordings
    dates = []
    for date in paths:
        dates.append(os.path.basename(date)[11:21])

    # Hour of the recordings
    heures = []
    for heure in paths:
        heures.append(os.path.basename(heure)[22:-4])

    # DataFrame with all the info
    metadata = pd.DataFrame(zip(paths, dates, heures), columns=['fd', 'day', 'hour'])

    # Suppression of nighty recordings
    indexNames = metadata[(metadata['hour'] == '00h-01h') |
                          (metadata['hour'] == '01h-02h') |
                          (metadata['hour'] == '02h-03h') |
                          (metadata['hour'] == '03h-04h') |
                          (metadata['hour'] == '04h-05h') |
                          (metadata['hour'] == '05h-06h') |
                          (metadata['hour'] == '06h-07h') |
                          (metadata['hour'] == '21h-22h') |
                          (metadata['hour'] == '22h-23h') |
                          (metadata['hour'] == '23h-24h')].index
    metadata.drop(indexNames, inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    if length is True:
        # tf df with the info
        ds = tf.data.Dataset.from_tensor_slices((metadata["fd"], metadata["day"], metadata["hour"]))

        # Function to load the recordings
        def get_waveform(fd, day, hour):
            audio_binary = tf.io.read_file(fd)
            audio, sr_in = tf.audio.decode_wav(audio_binary)
            audio = tfio.audio.resample(input=audio,
                                        rate_in=tf.cast(sr_in, tf.int64),
                                        rate_out=16000)
            audio = tf.reduce_mean(audio, axis=1)
            length = tf.shape(audio)
            frames = tf.signal.frame(audio, frame_length=16000,
                                     frame_step=3200,
                                     pad_end=True, pad_value=0, axis=-1)
            n_frames = tf.shape(frames)[0]

            return length, day, hour, n_frames, fd

        ds = ds.map(get_waveform).apply(tf.data.experimental.ignore_errors())

        len_files = []
        day_files = []
        hour_files = []
        frames_files = []
        fd_files = []
        for length, day, hour, n_frames, fd in ds:
            len_files.append(length.numpy()[0])
            day_files.append(day.numpy())
            hour_files.append(hour.numpy())
            frames_files.append(n_frames.numpy())
            fd_files.append(fd.numpy())

        day_files = [x.decode("utf-8") for x in day_files]
        hour_files = [x.decode("utf-8") for x in hour_files]
        fd_files = [x.decode("utf-8") for x in fd_files]

        meta_len = pd.DataFrame({'len': len_files, 'day': day_files, 'hour': hour_files, 'frames': frames_files, 'fd': fd_files})

        metadata = pd.merge(metadata, meta_len, on=["day", "hour", "fd"], how="inner")

    else:
        pass

    return metadata


def metadata_longform_baby(path="/home/guillhem/pCloudDrive/Documents/babyvoc/enregistrements", length=False):
    """
    Metadata of the long form audio recordings. Create a dataframe with as many
    rows as recordings, with three variables for each: the path, the children
    recorded and the date of the recordings.

    Parameters
    ----------
    path : string
        Path to go to the file where the recordings are.
    length : bool
        If True, load all the files to compute their length and the number of
        frames.

    Returns
    -------
    metadata : pd.DataFrame(n, 3)
        DataFrame
    """
    # List of the set of the children recorded
    children = np.array(os.listdir(path))
    # Creation of the dataframe
    meta = pd.DataFrame(columns=["child", "files"])
    # For each child
    for child in children:
        # List of the wav files
        files = [a for a in os.listdir(os.path.join(path, child, "fixed")) if "wav" in a]
        # we keep in a temporary df for each child the set of the recordings
        tmp = pd.DataFrame({"enfant": child, "files": files})
        # we merge all the reults in the global df
        meta = pd.concat([meta, tmp])

    # Variable to have direct access to the file of the recordings
    meta["fd"] = path + "/" + meta["child"] + "/fixed/" + meta["files"]
    # On remet les indices à 0.
    meta.reset_index(inplace=True)

    # Creation of a date variable tractable by pandas. All the infos is in the
    # name of the file. Il file is too long, it has been cut into several files.
    # These files have longer names. We need to take care of that when we take
    # the information about the date from the date of the file.
    for index, row in meta.iterrows():
        if len(row["files"]) == 17:
            # suppression of the file extension
            meta.loc[index, "date"] = row["files"][:-4]
        else:
            # suppression of 000n.wav, n corresponding to the partition.
            meta.loc[index, "date"] = row["files"][:-9]

    # we add the information about the millenary, necessary for pandas but not
    # present in the original recordings
    meta["date"] = "20" + meta["date"]
    # We transform the variable into temporal variable
    meta["date"] = pd.to_datetime(meta["date"], format="%Y%m%d-%H%M%S")

    if length is True:
        # Tf dataframe to load the recordings
        # To note, error trying to import a type datetime64. We need first to
        # convert the variable into string.
        ds = tf.data.Dataset.from_tensor_slices((meta["fd"], meta["date"].astype(str), meta["child"]))

        def get_waveform(fd, date, child):
            audio_binary = tf.io.read_file(fd)
            audio, sr_in = tf.audio.decode_wav(audio_binary)
            audio = tfio.audio.resample(input=audio,
                                        rate_in=tf.cast(sr_in, tf.int64),
                                        rate_out=16000)
            audio = tf.reduce_mean(audio, axis=1)
            length = tf.shape(audio)
            frames = tf.signal.frame(audio, frame_length=16000,
                                     frame_step=3200, pad_end=True,
                                     pad_value=0, axis=-1)
            n_frames = tf.shape(frames)[0]

            return length, date, child, n_frames, fd

        ds = ds.map(get_waveform).apply(tf.data.experimental.ignore_errors())

        len_files = []
        dates_files = []
        child_files = []
        frames_files = []
        fd_files = []
        for length, date, child, n_frames, fd in ds:
            len_files.append(length.numpy()[0])
            dates_files.append(date.numpy())
            child_files.append(child.numpy())
            frames_files.append(n_frames.numpy())
            fd_files.append(fd.numpy())

        dates_files = [x.decode("utf-8") for x in dates_files]
        child_files = [x.decode("utf-8") for x in child_files]
        fd_files = [x.decode("utf-8") for x in fd_files]

        meta_len = pd.DataFrame({'len': len_files, 'date': pd.to_datetime(dates_files), 'child': child_files, 'frames': frames_files, 'fd': fd_files})

        meta = pd.merge(meta, meta_len, on=["date", "child", "fd"], how="inner")

    else:
        pass

    return meta

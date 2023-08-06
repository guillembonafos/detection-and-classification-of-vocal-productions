import os
import muda
import jams

#####################
# Data augmentation #
#####################


def augmentation_data(metadata, wav_noise, path_saved, n_noise=1,
                      n_tonality=5, low_tonality=-1, high_tonality=1,
                      n_speed=5, low_speed=-0.3, high_speed=0.3):
    """
    Augmentation of the training set by modification the original recordings.
    Following McFee et al. (2015), three effects can be applied: adding background
    noise, on the tonality (increase/decrease), on the speed (acceleration/decelaration).
    Need to prepare the directory where the files will be created (in path_saved).
    path_saved must contain three folders: noise, tonality and speed.
    Utilisation of the library muda for the transformations.

    Parameters
    ----------
    metadata : pd.Dataframe
        DataFrame with the metadata of the original audio file recordings. Must
        contain a column name fd with the absolute path to the file.
    wav_noise : list
        List of strings, paths to the records without signal which are used as
        backgroung noise. To each recordings in the list, n_noise supplementary
        files are created.
    path_saved : basestring
        Path where save the created files.
    n_noise : int
        Number of new recordings to create with background noise from the original
        records. n_noise * number of elements in the list wav_noise.
        Per default, 1.
    n_tonality :int
        Number of new recordings to create changing the tonality of the original
        records. Changes between low_tonality and high_tonality.
        Per default, 5.
    low_tonality : float
        Inf boundary of the modification of the tonality.
        Per default, -1.
    high_tonality : float
        Sup boundary of the modification of the tonality.
        Per default, 1.
    n_speed : int
        Number of new recordings to create changing the speed of the original
        records. Changes between low_speed and high_speed.
        Per default, 5.
    low_speed : float
        Inf boundary of the modification of the speed.
        Per default, -0.3.
    high_speed : float
        Max boundary of the modification of the speed.
        Per default, 0.3.

    Returns
    -------
    Create new recordings, modifications from the originals contained in metadata,
    in the path_saved. The number of new files created depend on the quantity specified
    in the arguments of the functions, per default, 5 with a new tonality, 5 with
    a different speed, n(wav_noise) with background noise.
    """
    # e define the different effects
    if n_speed != 0:
        speed = muda.deformers.LogspaceTimeStretch(n_samples=n_speed, lower=low_speed, upper=high_speed)
    else:
        pass
    if n_tonality != 0:
        tonality = muda.deformers.LinearPitchShift(n_samples=n_tonality, lower=low_tonality, upper=high_tonality)
    else:
        pass
    if n_noise != 0:
        noise = muda.deformers.BackgroundNoise(n_samples=n_noise, files=wav_noise)
    else:
        pass

    for index, row in metadata.dropna().iterrows():
        # We load the original recordings through a jam file
        j_orig = muda.load_jam_audio(jams.JAMS(), row['fd'])

        # On save the new recordings after application of the effect
        if n_speed != 0:
            for i, vit in enumerate(speed.transform(j_orig)):
                fd = os.path.join(path_saved, 'speed', str(os.path.basename(row['fd'])))
                muda.save(fd[:-4] + '-' + str(i) + '.wav',
                          path_saved + '/speed_{:02d}.jams'.format(index),
                          vit)
        else:
            pass
        if n_tonality != 0:
            for i, ton in enumerate(tonality.transform(j_orig)):
                fd = os.path.join(path_saved, 'tonality', str(os.path.basename(row['fd'])))
                muda.save(fd[:-4] + '-' + str(i) + '.wav',
                          path_saved + '/tonality_{:02d}.jams'.format(index),
                          ton)
        else:
            pass
        if n_noise != 0:
            for i, b in enumerate(noise.transform(j_orig)):
                fd = os.path.join(path_saved, 'noise', str(os.path.basename(row['fd'])))
                muda.save(fd[:-4] + '-' + str(i) + '.wav',
                          path_saved + '/noise_{:02d}.jams'.format(index),
                          b)
        else:
            pass

        # We print the displayed
        print(index)

    return

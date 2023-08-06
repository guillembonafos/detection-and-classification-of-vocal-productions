import portion as P
import numpy as np
import os
from pydub import AudioSegment
import re
import pandas as pd


def segmentation(meta, y, baby=False):
    """
    Outputs txt files from the prediction that the model did on the long form
    audio recordings. On that files is written, for each recordings, the vocalizations
    detected, the moment they start and they end. These files are then used to
    extract the vocalizations into separated wav files.
    For the baby case, the txt files will be saved on ./data/segmentation_baby/{identification_of_the_baby}/{date}.txt,
    one file per recordings.
    For the baboon case, the txt files will be saved on ./data/segmentation_papio/{day}_{hour}.txt,
    one file per recordings.
    For the prediction of each frame, the previous (and the next) can be in three
    different situations: overlap, at less than one second, at more than one second.
    Because we merge the vocalizations of less than one seconds into a single one
    (and the overlapping frame are the same vocalization), there are 9 possible
    combinations, that we can reduct to 4: or a new vocalization starts, or a
    vocalization ends, or a ne vocalization starts and ends (vocalization of one
    second) or intermediary frame of a vocalization which has already started but
    is not yet over.

    Parameters
    ----------
    meta : pd.DataFrame
        DataFrame with the metadata of the long form audio recordings in which
        we do the segmentation. For the baby case, must contain a column named
        child with the identificator of the child recorded, a column named date
        with the date of the recordings, a column named frames with the number
        of frames in the recordidngs. The function metadata_longform_baby with
        length=True from scripts.functions.metadata produces an adapted df.
        For the papio case, must contain a column named day with the day of the
        recordings, a column hour with the hour of the recordings, a column frames
        with the number of frames of the recordings. The function metadata_longform_papio
        with length=True from scripts.functions.metadata produces an adapted df.
    y : list
        List of two arrays, outputs of the model on the long form audio recordings

    Returns
    -------
    The txt files used to do the extraction of the vocalization from the long form
    audio recordings. It resume each vocalization found in each long form recordings,
    with the beginning and the end of each vocalization predicted in the audio,
    in seconds.
    """
    n_frames = 0
    for k in range(meta.shape[0]):
        if baby is True:
            file = open(os.path.join(os.getcwd(), 'data/segmentation_baby/') + meta.loc[k, "child"] + "/" + meta.loc[k, "date"][:-4] + ".txt", "w")
        else:
            file = open(os.path.join(os.getcwd(), 'data/segmentation_papio/') + meta.loc[k, "day"] + '_' + meta.loc[k, "hour"] + '.txt', "w")
        # Computing the moment of the prediction (in time), given its position
        # in the recordings
        # First, construction of an arange corresponding to the number of frames
        # of the recordings
        tl_hour = np.arange(meta.loc[k, "frames"])
        # Indices of frames where there is signal. It is the information that
        # will be used to construct the txt file for the segmentation
        indices_hour = tl_hour[y[0].reshape(-1)[n_frames:n_frames + meta.loc[k, "frames"]] > 0.5]
        # We add the frames to select the correct predictions at the next recordgins
        # at the next iteration
        n_frames += meta.loc[k, "frames"]
        print(n_frames)
        # We start to count on the predictions
        nbr_vocs = 0
        # Special case when there is just one positive frame.
        if len(indices_hour) == 1:
            debut_voc = indices_hour[0]
            nbr_vocs = 1
            file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * debut_voc + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
            print(nbr_vocs)
            file.close()
        # Other case
        else:
            for i in range(len(indices_hour)):
                print(i)
                ##############
                # First bloc #
                ##############
                if i == 0:
                    # Case 1 :
                    # the end of block i take into account block i+1 or the block
                    # i+1 is at less that one second, i.e., block 2 is part of
                    # voc. The voc is more than one second.
                    if P.closed(np.round(0.2 * indices_hour[i + 1], 2),
                                np.round(0.2 * indices_hour[i + 1] + 1, 2)).overlaps(
                        P.closed(np.round(indices_hour[i] * 0.2, 2),
                                 np.round(indices_hour[i] * 0.2 + 1, 2))) or \
                            P.closed(np.round(0.2 * indices_hour[i + 1], 2),
                                     np.round(0.2 * indices_hour[i + 1] + 1, 2)).overlaps(
                                P.closed(np.round(0.2 * indices_hour[i] + 1, 2),
                                         np.round(0.2 * indices_hour[i] + 2, 2))):
                        debut_voc = indices_hour[i]
                    # Case 2 :
                    # the end of block i does not take into account block i+1 and
                    # block i+1 is at more than one second from the block i, i.e.,
                    # voc of one second et the next is too far away to be merge with
                    else:
                        debut_voc = indices_hour[i]
                        nbr_vocs += 1
                        file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * indices_hour[i] + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
                #######################
                # Intermediary blocks #
                #######################
                elif i != 0 and i != (len(indices_hour) - 1):
                    # Case 1:
                    # A new vocalization starts and continues after. It concerns
                    # to possibilities: (the previous is at more than one second
                    # AND the next is overlapped) OR (the previous is at more than
                    # one second AND the next is at less than one second)
                    if (0.2 * indices_hour[i - 1] + 2 < 0.2 * indices_hour[i] and
                        P.closed(0.2 * indices_hour[i + 1], 0.2 * indices_hour[i + 1] + 1).overlaps(
                            P.closed(0.2 * indices_hour[i], 0.2 * indices_hour[i] + 1))) or \
                            (0.2 * indices_hour[i - 1] + 2 < 0.2 * indices_hour[i] and
                                 P.closed(0.2 * indices_hour[i + 1], 0.2 * indices_hour[i + 1] + 1).overlaps(
                                     P.closed(0.2 * indices_hour[i] + 1, 0.2 * indices_hour[i] + 2))):
                        debut_voc = indices_hour[i]
                    # Case 2 :
                    # The vocalization starts and stops now. Vocalization of one
                    # second. It concerns one situation: the previous is at more
                    # than one second AND the next is at more than one second.
                    elif (0.2 * indices_hour[i - 1] + 2 < 0.2 * indices_hour[i]) and (0.2 * indices_hour[i + 1] > 0.2 * indices_hour[i] + 2):
                        debut_voc = indices_hour[i]
                        nbr_vocs += 1
                        file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * indices_hour[i] + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
                    # Case 3 :
                    # The vocalization has already started and end now. It concerns
                    # two possibilities: (the previous overlap the current AND
                    # the next is at more than one second) OR (the previous is
                    # at less than one second AND the next is at more than one
                    # second)
                    elif (P.closed(0.2 * indices_hour[i - 1], 0.2 * indices_hour[i - 1] + 1).overlaps(
                            P.closed(0.2 * indices_hour[i], 0.2 * indices_hour[i] + 1)) and
                          (0.2 * indices_hour[i + 1] > 0.2 * indices_hour[i] + 2)) or \
                            (P.closed(0.2 * indices_hour[i - 1], 0.2 * indices_hour[i - 1] + 2).overlaps(
                                P.closed(0.2 * indices_hour[i], 0.2 * indices_hour[i] + 1)) and
                             (0.2 * indices_hour[i + 1] > 0.2 * indices_hour[i] + 2)):
                        nbr_vocs += 1
                        file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * indices_hour[i] + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
                    # Case 4:
                    # No starting vocalization, it has already started in a previous
                    # frame and does not end now, it continues after.
                    # It concerns four possibilites: (the previous overlaps the
                    # current AND the next is overlapped) OR (the previous overlaps
                    # the current AND the next is at less than one second) OR (the
                    # previous is at less than one second AND the next is overlapped)
                    # OR (the previous is at less than one second AND the next is
                    # at less than one second)
                    else:
                        pass
                #################
                # Le last block #
                #################
                elif i == (len(indices_hour) - 1):
                    # Case 1:
                    # The end of block i-1 takes into account block i or is at less
                    # than one second, i.e., voc of more than one second which
                    # has started before
                    if P.closed(0.2 * indices_hour[i - 1], 0.2 * indices_hour[i] + 1).overlaps(
                            P.closed(0.2 * indices_hour[i], 0.2 * indices_hour[i] + 1)) or \
                            P.closed(0.2 * indices_hour[i - 1], 0.2 * indices_hour[i - 1] + 2).overlaps(
                                P.closed(0.2 * indices_hour[i], 0.2 * indices_hour[i] + 1)):
                        nbr_vocs += 1
                        file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * indices_hour[i] + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
                    # Case 2 :
                    # The end of block i-1 does not take into account block i,
                    # i.e., voc of one second which starts and ends at this frame
                    else:
                        debut_voc = indices_hour[i]
                        nbr_vocs += 1
                        file.write(str(round(0.2 * debut_voc, 2)) + "\t" + str(round(0.2 * indices_hour[i] + 1, 2)) + "\tvoc_" + str(nbr_vocs) + "\n")
                print(nbr_vocs)
            file.close()


def wav_creation(fd, baby=False):
    """
    Extraction of the segments found by the neural network. The function segmentation
    has to be run before, to produce the txt files resuming the beginning and the
    end of each vocalization, for each audio files. The function segmentation has
    to be run in the same working directory as wav_creation, to be able to find
    the txt files.

    Parameters
    ----------
    fd: basestring
        Path to find the long form audio recordings
    baby : bool
        If True, baby recordings. If False, papio recordings.
        Take into account the particular path of each case.

    Returns
    -------
    Create one folder per recordings, with as many file as vocalization detected
    by the model. Each file is a wave file extracted from the original long form
    audio recordings.
    """
    # Baby case
    if baby is True:
        #fd = "/home/guillhem/pCloudDrive/Documents/babyvoc/enregistrements"
        # List of folders, one per child
        enfants = [f for f in os.listdir(os.path.join(os.getcwd(), "data/segmentation_baby/"))]
        enfants.sort()
        # For each child
        for enfant in enfants:
            # List of file, one per recordings
            seg_files = [f for f in os.listdir(os.path.join(os.getcwd(), "data/segmentation_baby", enfant))]
            seg_files.sort()

            # For each recordings
            for i in range(len(seg_files)):
                # Creation of a new directory, to put the vocalizations extracted
                # from the recordings. The directories are created in the same
                # place where the txt files of segmentation were saved. The segments
                # are saved in each directory, one per recordings
                os.mkdir(os.path.join(os.getcwd(), "data/segmentation_baby", enfant, seg_files[i][:-4]))
                # We read the file
                vocs = []
                with open(os.path.join(os.getcwd(), "data/segmentation_baby", enfant, seg_files[i])) as myfile:
                    for line in myfile:
                        vocs.append(re.split(r'\t+', line))

                # We save the information in a df
                df = pd.DataFrame(vocs, columns=["start", "end", "voc"])
                df["voc"] = df["voc"].map(lambda x: str(x)[:-1])

                # We load the wav
                wav = AudioSegment.from_file(os.path.join(fd, enfant, "fixed", seg_files[i][:-4] + ".wav"))

                for index, row in df.iterrows():
                    # We build the new audio
                    # pydub works in milliseconds
                    start = float(df.loc[index, "start"]) * 1000
                    end = float(df.loc[index, "end"]) * 1000
                    # We select the part we are looking for
                    new_wav = wav[start:end]
                    # We export the wav segmented files
                    new_wav.export(os.path.join(os.getcwd(), "data/segmentation_baby/", enfant, seg_files[i][:-4], df.loc[index, "voc"] + ".wav"),
                                       format="wav")

                print(i)

    # Papio case
    else:
        #fd = "/media/guillhem/BackUParnaud/LOG_DATA"
        seg_files = [f for f in os.listdir(os.path.join(os.getcwd(), "data/segmentation_papio/"))]
        seg_files.sort()

        for i in range(len(seg_files)):
            # We create the directory where we will save the segments to extract
            # from the long form audio recordings
            os.mkdir(os.path.join(os.getcwd(), "data/segmentation_papio/", seg_files[i][:-4]))

            # We read the txt files with the information about the segmentation
            vocs = []
            with open(os.path.join(os.getcwd(), "data/segmentation_papio/", seg_files[i])) as myfile:
                for line in myfile:
                    vocs.append(re.split(r'\t+', line))

            # We save the info in a df
            df = pd.DataFrame(vocs, columns=["start", "end", "voc"])
            df["voc"] = df["voc"].map(lambda x: str(x)[:-1])

            # We load the corresponding wawv
            wav = AudioSegment.from_file(os.path.join(fd, "AudioRec01_" + seg_files[i][:-4] + ".wav"))

            for index, row in df.iterrows():
                # We build the audio
                start = float(df.loc[index, "start"]) * 1000
                end = float(df.loc[index, "end"]) * 1000
                # We select the segment
                new_wav = wav[start:end]
                # We export the vocalization
                new_wav.export(os.path.join(os.getcwd(), "data/segmentation_papio/", seg_files[i][:-4], df.loc[index, "voc"] + ".wav"), format="wav")

            print(i)


def df_pred(meta, y, bebe=False):
    """
    Construction of the datafram which resume the info of the segments detected.
    It is composed for each vocalization, by:
        -the label;
        -the recordings it is extracted from;
        -the beginning in the recordings;
        -the end in the recordings;
        -the duration;
        -the date (%d-%m-%Y%H:%M:%S);
        -the mean of the prediction the model did on the frames that composed
        the vocalization, for each possible label (sum to 1, each one =>0);
        -the number of frames the model predicted as containing the label, for
        each possible label (sum to the number of frames composing the vocalization).
    For the baby case, there is also the indicator of the baby.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata of the long form audio recordings on which the model did the
        prediction. Must contain the length of the recordings and the number of
        frames it contains.
        The df outputs by the function metadata_longform_baby and metadata_longform_papio
        are fitted to. The segmentation function should have been run before (to
        produce the txt files. Not necessary the wav_creation function). The
        file tree must be organized as previously (txt files in ./data/segmentation_papio
        for the papio case, in ./data/segmentation_baby/{child_identificator}
        for the children case)
    y : list
        List of two arrays, outputs of the model on the long form audio recordings

    Returns
    -------
    df : pd.DataFrame
        DataFrame with the information on the vocalizations detected by the model.
        One row per vocalization.
    """
    if bebe is True:
        # List of directories, one per child
        enfants = [f for f in os.listdir(os.path.join(os.getcwd(), "data/segmentation_baby/"))]
        enfants.sort()

        # In y, the prediction are done per frame. Each frame follows the previous,
        # is one second long and has an 80% overlap.
        # We have in the meta df the total number of frame per recordings to take
        # the count and have the correct correspondance.
        n_frames = 0
        # Df to save the results
        df = pd.DataFrame(
            columns=["type_voc", "file", "debut_voc", "fin_voc", "duree_voc",
                     "p_0", "p_1", "p_2", "p_3", "p_4", "date", "enfant",
                     "n_0", "n_1", "n_2", "n_3", "n_4", "voc"])

        # We add the variable seg_file, which is the txt file with the segmentation
        # to have all the info in the df
        # The function segmentation must have been run before to produce the txt
        # files
        meta["seg_file"] = meta["files"].map(lambda x: str(x)[:-4] + ".txt")

        for enfant in enfants:
            # seg_files list not useful thanks to the previous code
            # Info on the segmentation
            seg_files = [f for f in os.listdir(os.path.join(os.getcwd(), "data/segmentation_baby/", enfant)) if os.path.isfile(os.path.join(os.getcwd(), "data/segmentation_baby/", enfant, f))]
            # sort to having the correspondance with the meta df
            seg_files.sort()

            # We take the subset of the metadata corresponding to the child and
            # we reset indexes for iteration
            meta_tmp = meta[meta["child"] == enfant]
            meta_tmp.reset_index(drop=True, inplace=True)

            # Iteration for each recordings
            for i in range(len(meta_tmp)):
                # Beginning and end of each vocalization
                vocs = []
                with open(os.path.join(os.getcwd(), "data/segmentation_baby/", enfant,
                                       meta_tmp.loc[i, "seg_file"])) as myfile:
                    for line in myfile:
                        vocs.append(re.split(r'\t+', line))

                # Summary of info in a df
                df_voc = pd.DataFrame(vocs, columns=["start", "end", "voc"])
                df_voc["voc"] = df_voc["voc"].map(lambda x: str(x)[:-1])

                # Creation of vectors of length of the number of voc detected
                type_voc = np.zeros(len(vocs))
                p = np.zeros((len(vocs), 5))
                n_v = np.zeros((len(vocs), 5))
                deb_v = np.zeros(len(vocs))
                fin_v = np.zeros(len(vocs))
                delta_v = np.zeros(len(vocs))
                voc_v = []

                # We compute, for each vocalization, the probability it is a
                # detected, the label predicted and the probability of each labels,
                # by taking the mean on the frames.
                # We compute by majority voting the label of the vocalization.
                # 1 second = 5 frames
                for k in range(len(vocs)):
                    # Start of voc k
                    debut_voc = float(vocs[k][0])
                    # End of voc k
                    fin_voc = float(vocs[k][1])
                    # duration of voc k
                    delta = round(fin_voc - debut_voc, 1)
                    # label of voc k
                    voc = vocs[k][2]
                    # delta: number of seconds between the beginning and the end
                    # of the voc. delta*5=> number of frames between the beginning
                    # and the end
                    # the end in seconds is the last frame, we need to remove 4
                    # frames overlapping the next second (without signal).
                    p[k,] = np.mean(y[1][int(debut_voc * 5 + n_frames):int(
                        debut_voc * 5 + n_frames + delta * 5 - 4)], axis=0)
                    n_v[k,] = np.bincount(np.argmax(y[1][int(
                        debut_voc * 5 + n_frames):int(
                        debut_voc * 5 + n_frames + delta * 5 - 4)], axis=1),
                                          minlength=5)
                    type_voc[k] = np.argmax(n_v[k,])
                    deb_v[k] = debut_voc
                    fin_v[k] = fin_voc
                    delta_v[k] = delta
                    voc_v.append(voc)

                # We keep the useful info in metadata, date and file.
                file = meta_tmp.loc[i, "files"]
                heure = meta_tmp.loc[i, "date"]

                # Df wih the info we are looking for, label, file, identificator
                # of the child, start, end and duration, mean probability each
                # label, vote for each label
                df_tmp = pd.DataFrame(
                    {"type_voc": type_voc, "file": np.tile(file, len(vocs)),
                     "enfant": np.tile(enfant, len(vocs)),
                     "debut_voc": deb_v, "fin_voc": fin_v, "duree_voc": delta_v,
                     "p_0": p[:, 0], "p_1": p[:, 1], "p_2": p[:, 2],
                     "p_3": p[:, 3], "p_4": p[:, 4],
                     "n_0": n_v[:, 0], "n_1": n_v[:, 1], "n_2": n_v[:, 2],
                     "n_3": n_v[:, 3], "n_4": n_v[:, 4], "voc": voc_v})

                # Creation of date variable, with temporal info
                df_tmp["date"] = pd.to_datetime(df_tmp["debut_voc"], unit="s",
                                                origin=str(heure)).round("L")

                # concatenation of the recordings with the previous
                df = pd.concat([df, df_tmp])

                # Incrementation of the number of frames
                n_frames += meta_tmp.loc[i, "frames"]
                print(i)

        df["voc"] = df["voc"].replace('\n','', regex=True)
        df.set_index(df["date"], inplace=True)


    # Papio case
    else:
        # Info on the segmentation. The function segmentation must have been run
        # before to produce the txt files
        seg_files = [f for f in os.listdir(
            os.path.join(os.getcwd(), "data/segmentation_papio/")) if
                     os.path.isfile(os.path.join(os.getcwd(),
                                                 "data/segmentation_papio/",
                                                 f))]
        # sort to having the correspondance with the meta df
        seg_files.sort()

        # We have in the meta df the total number of frame per recordings to take
        # the count and have the correct correspondance.
        n_frames = 0
        # Df to save the results
        df = pd.DataFrame(
            columns=["type_voc", "file", "debut_voc", "fin_voc", "duree_voc",
                     "p_0", "p_1", "p_2", "p_3", "p_4", "p_5", "date",
                     "n_0", "n_1", "n_2", "n_3", "n_4", "n_5", "voc"])
        # Iteration for each recordings
        for i in range(len(meta)):
            # Beginning and end of each vocalization
            vocs = []
            with open(os.path.join(os.getcwd(), "data/segmentation_papio/",
                                   seg_files[i])) as myfile:
                for line in myfile:
                    vocs.append(re.split(r'\t+', line))

            # Summary of info in a df
            df_voc = pd.DataFrame(vocs, columns=["start", "end", "voc"])
            df_voc["voc"] = df_voc["voc"].map(lambda x: str(x)[:-1])

            # Creation of vectors of length of the number of voc detected
            type_voc = np.zeros(len(vocs))
            p = np.zeros((len(vocs), 6))
            n_v = np.zeros((len(vocs), 6))
            deb_v = np.zeros(len(vocs))
            fin_v = np.zeros(len(vocs))
            delta_v = np.zeros(len(vocs))
            voc_v = []

            # We compute, for each vocalization, the probability it is a
            # detected, the label predicted and the probability of each labels,
            # by taking the mean on the frames.
            # We compute by majority voting the label of the vocalization.
            # 1 second = 5 frames
            for k in range(len(vocs)):
                # beginning of voc k
                debut_voc = float(vocs[k][0])
                # end of voc k
                fin_voc = float(vocs[k][1])
                # duration of voc k
                delta = round(fin_voc - debut_voc, 1)
                # label of voc k
                voc = vocs[k][2]
                # delta: number of seconds between the beginning and the end
                # of the voc. delta*5=> number of frames between the beginning
                # and the end
                # the end in seconds is the last frame, we need to remove 4
                # frames overlapping the next second (without signal).
                p[k,] = np.mean(y[1][int(debut_voc * 5 + n_frames):int(
                    debut_voc * 5 + n_frames + delta * 5 - 4)], axis=0)
                n_v[k,] = np.bincount(np.argmax(y[1][int(
                    debut_voc * 5 + n_frames):int(
                    debut_voc * 5 + n_frames + delta * 5 - 4)], axis=1),
                                      minlength=6)
                type_voc[k] = np.argmax(n_v[k,])
                deb_v[k] = debut_voc
                fin_v[k] = fin_voc
                delta_v[k] = delta
                voc_v.append(voc)

            # We create the variable which give the exact moment of the vocalization
            # (%d-%m-%Y%H:%M:%S). This information will be used for the creation
            # of the variable date in the final df.
            # We test if the name is correct to find the info (some recordings
            # last less than one hour)
            if len(meta.loc[i, "heure"]) == 7:
                heure = pd.to_datetime(
                    str(meta.loc[i, "jour"] + meta.loc[i, "heure"][0:2]),
                    format="%d-%m-%Y%H")
                file = meta.loc[i, "jour"] + '_' + meta.loc[i, "heure"]
            else:
                heure = pd.to_datetime(str(
                    meta.loc[i, "jour"] + meta.loc[i, "heure"][0:2] + ":" +
                    meta.loc[i, "heure"][3:5] + ":" + meta.loc[i, "heure"][
                                                      6:8]),
                                       format="%d-%m-%Y%H:%M:%S")
                file = meta.loc[i, "jour"] + '_' + meta.loc[i, "heure"]

            # Dataframe avec les infos dont on a besoin : le type de voc, sa durée, la probabilité
            # moyenne de chaque voc, le moment et le jour, le vote pour chaque voc, son
            # début, sa fin
            # Df wih the info we are looking for, label, file, start, end and
            # duration, mean probability each label, vote for each label, date
            df_tmp = pd.DataFrame(
                {"type_voc": type_voc, "file": np.tile(file, len(vocs)),
                 "debut_voc": deb_v, "fin_voc": fin_v, "duree_voc": delta_v,
                 "p_0": p[:, 0], "p_1": p[:, 1], "p_2": p[:, 2],
                 "p_3": p[:, 3], "p_4": p[:, 4], "p_5": p[:, 5],
                 "n_0": n_v[:, 0], "n_1": n_v[:, 1], "n_2": n_v[:, 2],
                 "n_3": n_v[:, 3], "n_4": n_v[:, 4], "n_5": n_v[:, 5], "voc": voc_v})

            # Creation of the date variable, which contains all the temporal informations
            df_tmp["date"] = pd.to_datetime(df_tmp["debut_voc"], unit="s",
                                            origin=str(heure)).round("L")

            # Concatenation with the previous vocalizations
            df = pd.concat([df, df_tmp])

            # Incrementation of the number of frames
            n_frames += meta.loc[i, "frames"]
            print(i)

        df["voc"] = df["voc"].replace('\n', '', regex=True)
        df.set_index(df["date"], inplace=True)

    return df

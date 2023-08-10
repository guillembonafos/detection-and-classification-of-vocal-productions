---
jupyter:
  kernelspec:
    display_name: detection-and-classification-of-vocal-productions-main
    language: python
    name: detection-and-classification-of-vocal-productions-main
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.2
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .code execution_count="1"}
``` python
#import scripts.functions.preprocessing as pp
#from scripts.functions.metadata import metadata_longform_papio, meta_papio
#from scripts.functions.hypermodel import WinWavTransferLearning
#from scripts.functions.segmentation import segmentation, df_pred, wav_creation
import os
import tensorflow as tf
import kerastuner as kt
import pandas as pd
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```
:::

::: {.cell .code execution_count="2"}
``` python
%run ./scripts/functions/preprocessing.py
%run ./scripts/functions/metadata.py
%run ./scripts/functions/hypermodel.py
%run ./scripts/functions/segmentation.py
%run ./scripts/functions/data_augmentation.py
```
:::

::: {.cell .markdown}
We present here an exemple to replicate the model used in the paper
<https://arxiv.org/abs/2302.07640>

The labeled data, as well as the longform audio recordings used as
exemple, and all the vocalizations extracted during the month, are
available at <https://doi.org/10.5281/zenodo.7963124>

The folder structure should be as follows: in the working directory,

-   a folder *data*. In this folder,
    -   a folder *vocalizations*, in which we have the labeled data,
    -   a folder *data_augmentation*, in which we save the modified
        vocalizations,
        -   a folder *test*, for the modified recordings of the test
            partition,
        -   a folder *train*, for the modified recordings of the train
            partition,
        -   a folder *val*, for the modified recordings of the
            validation partition.\
            For each one, there are three folders, one per condition of
            modification of the original recordings:
            -   a folder *noise*,
            -   a folder *tonality*,
            -   a folder *speed*,
    -   a folder *longform_recordings*, in which we have the long form
        audio recordings,
    -   a folder *segmentation_papio*, in which we will save the
        vocalizations detected.
-   a folder *scripts*. In this folder,
    -   a folder *functions*, in which we have all the functions used
        during the pipeline
    -   a folder *learning*, for the scripts used for the training of
        the models presentated in the paper,
    -   a folder *prediction*, for the scripts used for the prediction
        of all the long form audio recordings presented in the paper,
        that ouputs the data of the baboon and baby vocalizations.

The gitlab repository is already organized that way. The data available
on zenodo are organized that way. The libraries to install can be found
in the document requirements.txt

We present here an exemple to replicate the model using a subset of the
baboon recordings. We cannot provide all the month for legal reasons. We
put 2 hours accessible, as well as the labeled data set used in the
paper. We show how to train a model from the labeled dataset and how to
use it for the segmentation of these two hours.\
The total output of the segmentation of the month is available on
zenodo.

For the baby data, none of the long form audio recordings are available,
nor the output of the segmentation, for leagal reasons. The labeled
dataset is BabbleCor and can be found here <https://osf.io/rz4tx/>
:::

::: {.cell .markdown}
# Learning

We start loading the metadata of the labeled data for the learning.
:::

::: {.cell .code execution_count="3"}
``` python
meta_train, meta_val, meta_test = meta_papio(os.getcwd(), data_augmentation=False, weighting_sampling=False)
```
:::

::: {.cell .code execution_count="12"}
``` python
wav_noise = [os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_01-05-2017_09h-10h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_03-05-2017_11h-12h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_05-05-2017_16h-17h_sans_vocs.wav"),
            os.path.join(os.getcwd(), "data/recordings_without_vocs/AudioRec01_29-04-2017_18h-19h_sans_vocs.wav")]
```
:::

::: {.cell .code execution_count="16"}
``` python
augmentation_data(meta_train, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/train"),
                 wav_noise=wav_noise, n_noise=1,
                 n_tonality=5, low_tonality=-4, high_tonality=4,
                 n_speed=5, low_speed=-0.3, high_speed=0.3)
augmentation_data(meta_val, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/val"),
                 wav_noise=wav_noise, n_noise=1,
                 n_tonality=5, low_tonality=-4, high_tonality=4,
                 n_speed=5, low_speed=-0.3, high_speed=0.3)
```

::: {.output .error ename="RuntimeError" evalue="Failed to execute rubberband. Please verify that rubberband-cli is installed."}
    ---------------------------------------------------------------------------
    FileNotFoundError                         Traceback (most recent call last)
    ~/anaconda3/lib/python3.8/site-packages/pyrubberband/pyrb.py in __rubberband(y, sr, **kwargs)
         73 
    ---> 74         subprocess.check_call(arguments, stdout=DEVNULL, stderr=DEVNULL)
         75 

    ~/anaconda3/lib/python3.8/subprocess.py in check_call(*popenargs, **kwargs)
        358     """
    --> 359     retcode = call(*popenargs, **kwargs)
        360     if retcode:

    ~/anaconda3/lib/python3.8/subprocess.py in call(timeout, *popenargs, **kwargs)
        339     """
    --> 340     with Popen(*popenargs, **kwargs) as p:
        341         try:

    ~/anaconda3/lib/python3.8/subprocess.py in __init__(self, args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell, cwd, env, universal_newlines, startupinfo, creationflags, restore_signals, start_new_session, pass_fds, encoding, errors, text)
        853 
    --> 854             self._execute_child(args, executable, preexec_fn, close_fds,
        855                                 pass_fds, cwd, env,

    ~/anaconda3/lib/python3.8/subprocess.py in _execute_child(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, env, startupinfo, creationflags, shell, p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite, restore_signals, start_new_session)
       1701                         err_msg = os.strerror(errno_num)
    -> 1702                     raise child_exception_type(errno_num, err_msg, err_filename)
       1703                 raise child_exception_type(err_msg)

    FileNotFoundError: [Errno 2] No such file or directory: 'rubberband'

    The above exception was the direct cause of the following exception:

    RuntimeError                              Traceback (most recent call last)
    <ipython-input-16-b8d344914825> in <module>
    ----> 1 augmentation_data(meta_train, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/train"),
          2                  wav_noise=wav_noise, n_noise=1,
          3                  n_tonality=5, low_tonality=-4, high_tonality=4,
          4                  n_speed=5, low_speed=-0.3, high_speed=0.3)
          5 augmentation_data(meta_val, path_saved=os.path.join(os.getcwd(), "data/data_augmentation/val"),

    ~/Documents/detection-and-classification-of-vocal-productions-main/scripts/functions/data_augmentation.py in augmentation_data(metadata, wav_noise, path_saved, n_noise, n_tonality, low_tonality, high_tonality, n_speed, low_speed, high_speed)
         82         # On save the new recordings after application of the effect
         83         if n_speed != 0:
    ---> 84             for i, vit in enumerate(speed.transform(j_orig)):
         85                 fd = os.path.join(path_saved, 'speed', str(os.path.basename(row['fd'])))
         86                 muda.save(fd[:-4] + '-' + str(i) + '.wav',

    ~/anaconda3/lib/python3.8/site-packages/muda/base.py in transform(self, jam)
        157 
        158         for state in self.states(jam):
    --> 159             yield self._transform(jam, state)
        160 
        161     @property

    ~/anaconda3/lib/python3.8/site-packages/muda/base.py in _transform(self, jam, state)
        121 
        122         try:
    --> 123             self.audio(jam_w.sandbox.muda, state)
        124         except NotImplementedError:
        125             pass

    ~/anaconda3/lib/python3.8/site-packages/muda/deformers/time.py in audio(mudabox, state)
         30     def audio(mudabox, state):
         31         # Deform the audio and metadata
    ---> 32         mudabox._audio["y"] = pyrb.time_stretch(
         33             mudabox._audio["y"], mudabox._audio["sr"], state["rate"]
         34         )

    ~/anaconda3/lib/python3.8/site-packages/pyrubberband/pyrb.py in time_stretch(y, sr, rate, rbargs)
        140     rbargs.setdefault('--tempo', rate)
        141 
    --> 142     return __rubberband(y, sr, **rbargs)
        143 
        144 

    ~/anaconda3/lib/python3.8/site-packages/pyrubberband/pyrb.py in __rubberband(y, sr, **kwargs)
         82 
         83     except OSError as exc:
    ---> 84         six.raise_from(RuntimeError('Failed to execute rubberband. '
         85                                     'Please verify that rubberband-cli '
         86                                     'is installed.'),

    ~/anaconda3/lib/python3.8/site-packages/six.py in raise_from(value, from_value)

    RuntimeError: Failed to execute rubberband. Please verify that rubberband-cli is installed.
:::
:::

::: {.cell .markdown}
Then, we prepare the data. This is done through the creation of a
dataset, using the metadata of the labeled data. The recordings are
loaded and resampled at 16 KhZ, in mono. We create independant frames
through an 80% overlapping 1-second window. We use a resampling strategy
to have an uniform distribution among classes during the learning.
Because we use transfer-learning from YamNet, the frames are mapped to a
log-mel spectrogram. Data augmentation is done before, not during the
learning, because we do not expect to have so much labeled data. Thus,
we can gain time during the learning without being too expensive in term
of memory.
:::

::: {.cell .code execution_count="4"}
``` python
train, steps_per_epoch = preparation_data_set(meta_train, resample=True, batch_size=32, transfer_learning=True)
val = preparation_data_set(meta_val, resample=False, batch_size=32, transfer_learning=True)

input_shape = next(iter(train.unbatch()))[0].shape

train = train.shuffle(1000)
```

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:AutoGraph could not transform <function preparation_data_set.<locals>.equilibrage_classes.<locals>.<lambda> at 0x7fe3f9a46af0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: 'NoneType' object has no attribute '__dict__'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:AutoGraph could not transform <function preparation_data_set.<locals>.equilibrage_classes.<locals>.<lambda> at 0x7fe3f9a46af0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: 'NoneType' object has no attribute '__dict__'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
:::

::: {.output .stream .stdout}
    WARNING: AutoGraph could not transform <function preparation_data_set.<locals>.equilibrage_classes.<locals>.<lambda> at 0x7fe3f9a46af0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: 'NoneType' object has no attribute '__dict__'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING:tensorflow:From /home/guillem/anaconda3/lib/python3.8/site-packages/tensorflow/python/data/experimental/ops/interleave_ops.py:260: RandomDataset.__init__ (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.random(...)`.
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:From /home/guillem/anaconda3/lib/python3.8/site-packages/tensorflow/python/data/experimental/ops/interleave_ops.py:260: RandomDataset.__init__ (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.data.Dataset.random(...)`.
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::
:::

::: {.cell .markdown}
We instantiate the hypermodel and the hyperparameters, as well as the
callbacks. The values set here are for the exemple and can be changed
and increase for a \"true\" learning.
:::

::: {.cell .code execution_count="5"}
``` python
hypermodel = WinWavTransferLearning(input_shape=input_shape, n_labels=6)
hp = kt.HyperParameters()

tuner = kt.tuners.bayesian.BayesianOptimization(hypermodel=hypermodel,
                                               hyperparameters=hp,
                                               objective=kt.Objective("val_loss", direction="min"),
                                                # increase to have more searching iterations. 
                                                # Set to 2 here for the exemple
                                                max_trials=2,
                                                num_initial_points=1,
                                                tune_new_entries=True,
                                                project_name="exemple")

earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, 
                                             patience=5, verbose=1, 
                                             restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.getcwd(), "exemple/cp.hdf5"),
                                               monitor="val_loss", mode="min", 
                                                save_best_only=True, verbose=1)

history = tf.keras.callbacks.CSVLogger(os.path.join(os.getcwd(), "exemple/train.csv"),
                                      separator=",", append=False)


lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                                  patience=2, verbose=1)

callbacks = [earlystop, checkpoint, history, lr_schedule]
```

::: {.output .stream .stdout}
    INFO:tensorflow:Reloading Oracle from existing project ./exemple/oracle.json
:::

::: {.output .stream .stderr}
    INFO:tensorflow:Reloading Oracle from existing project ./exemple/oracle.json
:::

::: {.output .stream .stdout}
    INFO:tensorflow:Reloading Tuner from ./exemple/tuner0.json
:::

::: {.output .stream .stderr}
    INFO:tensorflow:Reloading Tuner from ./exemple/tuner0.json
:::
:::

::: {.cell .markdown}
We can start the learning.
:::

::: {.cell .code execution_count="21"}
``` python
# Disable AutoShard
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train = train.with_options(options)
val = val.with_options(options)

print("Start of the learning")
start = datetime.now()
tuner.search(train, epochs=20, validation_data=val, callbacks=callbacks,
            steps_per_epoch=steps_per_epoch)
delta_training = datetime.now() - start
```

::: {.output .stream .stdout}
    Trial 2 Complete [00h 40m 10s]
    val_loss: 0.11610516160726547

    Best val_loss So Far: 0.1083238422870636
    Total elapsed time: 01h 09m 21s
    INFO:tensorflow:Oracle triggered exit
:::

::: {.output .stream .stderr}
    INFO:tensorflow:Oracle triggered exit
:::
:::

::: {.cell .markdown}
# Prediction

The model has been learned on the labeled data and now can be used to
detect vocalizations in long form audio recordings. We start loading the
metadata of the longform audio recordings as well as their length and we
prepare the data to be processed by the model.
:::

::: {.cell .code execution_count="7"}
``` python
meta_longform = metadata_longform_papio(os.path.join(os.getcwd(), "data/longform_recordings"), 
                                       length=True)

ds = preparation_longform_papio(meta_longform, batch_size=32, transfer_learning=True)
```

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stdout}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::

::: {.output .stream .stderr}
    WARNING:tensorflow:Using a while_loop for converting IO>AudioResample
:::
:::

::: {.cell .markdown}
We take the best model of the optimization process and we use it to find
the segments of vocalizations in the recordings.
:::

::: {.cell .code execution_count="6"}
``` python
model = tuner.get_best_models()[0]
model.summary()
```

::: {.output .stream .stdout}
    Model: "model_classi"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_classification (InputLaye [(None, 1024)]       0                                            
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 1024)         4096        input_classification[0][0]       
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 171)          175275      batch_normalization[0][0]        
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 512)          524800      batch_normalization[0][0]        
    __________________________________________________________________________________________________
    p_re_lu_4 (PReLU)               (None, 171)          171         dense_4[0][0]                    
    __________________________________________________________________________________________________
    p_re_lu_3 (PReLU)               (None, 512)          512         dense_3[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, 171)          684         p_re_lu_4[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 512)          2048        p_re_lu_3[0][0]                  
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 171)          0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    dropout_3 (Dropout)             (None, 512)          0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    output_binaire (Dense)          (None, 1)            172         dropout_4[0][0]                  
    __________________________________________________________________________________________________
    output_multi (Dense)            (None, 6)            3078        dropout_3[0][0]                  
    ==================================================================================================
    Total params: 710,836
    Trainable params: 707,422
    Non-trainable params: 3,414
    __________________________________________________________________________________________________
:::
:::

::: {.cell .code execution_count="8"}
``` python
start = datetime.now()
y = model.predict(ds)
delta_pred = datetime.now() - start
print("Duration prediction:", delta_pred)
```

::: {.output .stream .stdout}
    Duration prediction: 0:02:40.343493
:::
:::

::: {.cell .markdown}
# Segmentation

Once we learned the model and used it to find the segments of
vocalizations in the longform audio recordings, we extract the
information. First, we create txt files, one per recordings, in which we
have the number of vocalizations found with the time in the recordings.
:::

::: {.cell .code execution_count="10"}
``` python
segmentation(meta_longform, y, baby=False)
```

::: {.output .stream .stdout}
    17998
    0
    1
    1
    1
    2
    1
    3
    1
    4
    1
    5
    2
    6
    2
    7
    2
    8
    3
    9
    3
    10
    3
    11
    3
    12
    3
    13
    3
    14
    3
    15
    3
    16
    3
    17
    3
    18
    3
    19
    3
    20
    3
    21
    4
    22
    4
    23
    5
    24
    6
    25
    6
    26
    6
    27
    6
    28
    6
    29
    6
    30
    6
    31
    7
    32
    7
    33
    7
    34
    7
    35
    7
    36
    7
    37
    7
    38
    7
    39
    7
    40
    7
    41
    7
    42
    7
    43
    7
    44
    7
    45
    8
    46
    8
    47
    9
    48
    10
    49
    10
    50
    10
    51
    10
    52
    11
    53
    11
    54
    11
    55
    11
    56
    11
    57
    11
    58
    11
    59
    11
    60
    11
    61
    11
    62
    11
    63
    11
    64
    11
    65
    11
    66
    11
    67
    11
    68
    11
    69
    11
    70
    11
    71
    11
    72
    11
    73
    11
    74
    11
    75
    11
    76
    11
    77
    11
    78
    11
    79
    11
    80
    11
    81
    11
    82
    12
    83
    12
    84
    12
    85
    12
    86
    12
    87
    12
    88
    12
    89
    12
    90
    12
    91
    12
    92
    12
    93
    12
    94
    12
    95
    12
    96
    13
    97
    13
    98
    13
    99
    13
    100
    13
    101
    14
    102
    14
    103
    15
    104
    16
    105
    17
    106
    17
    107
    17
    108
    17
    109
    17
    110
    17
    111
    17
    112
    17
    113
    17
    114
    17
    115
    17
    116
    17
    117
    17
    118
    17
    119
    17
    120
    17
    121
    17
    122
    17
    123
    17
    124
    17
    125
    17
    126
    17
    127
    17
    128
    17
    129
    17
    130
    17
    131
    17
    132
    17
    133
    17
    134
    17
    135
    17
    136
    17
    137
    17
    138
    17
    139
    17
    140
    17
    141
    17
    142
    17
    143
    18
    144
    18
    145
    18
    146
    18
    147
    18
    148
    19
    149
    19
    150
    19
    151
    20
    152
    20
    153
    20
    154
    20
    155
    20
    156
    21
    157
    21
    158
    21
    159
    21
    160
    21
    161
    21
    162
    21
    163
    21
    164
    21
    165
    21
    166
    21
    167
    21
    168
    21
    169
    21
    170
    21
    171
    21
    172
    21
    173
    21
    174
    21
    175
    21
    176
    21
    177
    21
    178
    21
    179
    21
    180
    21
    181
    21
    182
    21
    183
    21
    184
    21
    185
    21
    186
    21
    187
    21
    188
    21
    189
    21
    190
    21
    191
    21
    192
    21
    193
    21
    194
    21
    195
    21
    196
    21
    197
    21
    198
    21
    199
    21
    200
    21
    201
    21
    202
    21
    203
    21
    204
    21
    205
    21
    206
    21
    207
    21
    208
    21
    209
    21
    210
    21
    211
    21
    212
    21
    213
    21
    214
    21
    215
    21
    216
    21
    217
    21
    218
    21
    219
    21
    220
    21
    221
    21
    222
    21
    223
    21
    224
    21
    225
    21
    226
    21
    227
    21
    228
    21
    229
    21
    230
    21
    231
    21
    232
    21
    233
    21
    234
    21
    235
    21
    236
    21
    237
    21
    238
    21
    239
    21
    240
    21
    241
    21
    242
    21
    243
    21
    244
    21
    245
    21
    246
    21
    247
    21
    248
    21
    249
    21
    250
    21
    251
    21
    252
    21
    253
    21
    254
    21
    255
    21
    256
    21
    257
    21
    258
    21
    259
    21
    260
    21
    261
    21
    262
    21
    263
    21
    264
    21
    265
    21
    266
    21
    267
    21
    268
    21
    269
    21
    270
    21
    271
    21
    272
    21
    273
    21
    274
    21
    275
    21
    276
    21
    277
    21
    278
    21
    279
    21
    280
    21
    281
    21
    282
    21
    283
    21
    284
    21
    285
    21
    286
    21
    287
    21
    288
    21
    289
    21
    290
    21
    291
    21
    292
    21
    293
    21
    294
    22
    295
    22
    296
    22
    297
    22
    298
    22
    299
    22
    300
    22
    301
    22
    302
    22
    303
    22
    304
    22
    305
    22
    306
    22
    307
    22
    308
    22
    309
    22
    310
    22
    311
    22
    312
    22
    313
    22
    314
    22
    315
    22
    316
    22
    317
    22
    318
    22
    319
    22
    320
    22
    321
    22
    322
    22
    323
    22
    324
    22
    325
    22
    326
    22
    327
    22
    328
    22
    329
    22
    330
    22
    331
    22
    332
    22
    333
    22
    334
    22
    335
    22
    336
    22
    337
    22
    338
    22
    339
    22
    340
    22
    341
    22
    342
    22
    343
    22
    344
    22
    345
    22
    346
    22
    347
    22
    348
    22
    349
    22
    350
    22
    351
    22
    352
    22
    353
    22
    354
    22
    355
    22
    356
    22
    357
    22
    358
    22
    359
    22
    360
    22
    361
    22
    362
    22
    363
    22
    364
    22
    365
    22
    366
    22
    367
    22
    368
    22
    369
    22
    370
    22
    371
    22
    372
    22
    373
    22
    374
    22
    375
    22
    376
    22
    377
    22
    378
    22
    379
    22
    380
    22
    381
    22
    382
    22
    383
    22
    384
    22
    385
    22
    386
    22
    387
    22
    388
    22
    389
    22
    390
    22
    391
    22
    392
    22
    393
    22
    394
    22
    395
    22
    396
    22
    397
    22
    398
    22
    399
    22
    400
    22
    401
    22
    402
    22
    403
    22
    404
    22
    405
    22
    406
    22
    407
    22
    408
    22
    409
    22
    410
    22
    411
    22
    412
    22
    413
    22
    414
    22
    415
    22
    416
    22
    417
    22
    418
    22
    419
    22
    420
    22
    421
    22
    422
    22
    423
    22
    424
    22
    425
    22
    426
    22
    427
    22
    428
    22
    429
    22
    430
    22
    431
    23
    432
    23
    433
    23
    434
    23
    435
    23
    436
    23
    437
    24
    438
    24
    439
    24
    440
    24
    441
    24
    442
    24
    443
    24
    444
    24
    445
    24
    446
    24
    447
    24
    448
    24
    449
    24
    450
    24
    451
    24
    452
    24
    453
    24
    454
    24
    455
    24
    456
    24
    457
    24
    458
    24
    459
    25
    460
    25
    461
    25
    462
    25
    463
    25
    464
    25
    465
    25
    466
    25
    467
    25
    468
    25
    469
    25
    470
    25
    471
    25
    472
    25
    473
    25
    474
    25
    475
    25
    476
    25
    477
    25
    478
    25
    479
    25
    480
    25
    481
    25
    482
    25
    483
    25
    484
    25
    485
    25
    486
    25
    487
    25
    488
    25
    489
    25
    490
    25
    491
    25
    492
    25
    493
    25
    494
    25
    495
    25
    496
    25
    497
    25
    498
    25
    499
    25
    500
    25
    501
    25
    502
    25
    503
    25
    504
    25
    505
    25
    506
    25
    507
    25
    508
    25
    509
    25
    510
    25
    511
    25
    512
    25
    513
    25
    514
    26
    515
    26
    516
    26
    517
    26
    518
    26
    519
    26
    520
    26
    521
    26
    522
    26
    523
    26
    524
    26
    525
    26
    526
    26
    527
    26
    528
    26
    529
    26
    530
    26
    531
    26
    532
    26
    533
    26
    534
    26
    535
    26
    536
    26
    537
    26
    538
    26
    539
    26
    540
    26
    541
    26
    542
    26
    543
    26
    544
    26
    545
    26
    546
    26
    547
    26
    548
    26
    549
    26
    550
    26
    551
    26
    552
    26
    553
    26
    554
    26
    555
    27
    556
    27
    557
    27
    558
    27
    559
    27
    560
    27
    561
    27
    562
    27
    563
    27
    564
    27
    565
    28
    566
    28
    567
    29
    568
    29
    569
    30
    570
    30
    571
    31
    572
    32
    573
    33
    574
    33
    575
    33
    576
    33
    577
    33
    578
    33
    579
    33
    580
    33
    581
    34
    582
    34
    583
    34
    584
    34
    585
    34
    586
    34
    587
    34
    588
    34
    589
    34
    590
    34
    591
    34
    592
    34
    593
    34
    594
    34
    595
    35
    596
    36
    597
    36
    598
    36
    599
    36
    600
    37
    601
    37
    602
    38
    603
    38
    604
    38
    605
    38
    606
    39
    607
    39
    608
    39
    609
    40
    610
    40
    611
    40
    612
    40
    613
    40
    614
    40
    615
    40
    616
    40
    617
    40
    618
    40
    619
    40
    620
    40
    621
    40
    622
    40
    623
    40
    624
    40
    625
    40
    626
    41
    627
    41
    628
    41
    629
    41
    630
    41
    631
    41
    632
    41
    633
    41
    634
    41
    635
    42
    636
    43
    637
    44
    638
    44
    639
    44
    640
    44
    641
    44
    642
    44
    643
    44
    644
    44
    645
    44
    646
    44
    647
    44
    648
    44
    649
    44
    650
    44
    651
    44
    652
    44
    653
    44
    654
    44
    655
    44
    656
    44
    657
    44
    658
    44
    659
    44
    660
    44
    661
    44
    662
    44
    663
    44
    664
    44
    665
    44
    666
    44
    667
    44
    668
    44
    669
    44
    670
    44
    671
    44
    672
    44
    673
    44
    674
    44
    675
    44
    676
    44
    677
    44
    678
    44
    679
    44
    680
    44
    681
    44
    682
    45
    683
    46
    684
    46
    685
    47
    686
    47
    687
    47
    688
    47
    689
    47
    690
    47
    691
    47
    692
    47
    693
    47
    694
    47
    695
    48
    696
    48
    697
    48
    698
    48
    699
    48
    700
    48
    701
    48
    702
    48
    703
    48
    704
    48
    705
    48
    706
    48
    707
    48
    708
    48
    709
    48
    710
    49
    711
    50
    712
    50
    713
    50
    714
    51
    715
    51
    716
    51
    717
    51
    718
    52
    719
    53
    720
    53
    721
    53
    722
    53
    723
    53
    724
    53
    725
    54
    726
    54
    727
    55
    728
    55
    729
    55
    730
    55
    731
    55
    732
    55
    733
    55
    734
    55
    735
    56
    736
    57
    737
    57
    738
    58
    739
    58
    740
    58
    741
    58
    742
    58
    743
    58
    744
    58
    745
    58
    746
    58
    747
    58
    748
    58
    749
    58
    750
    58
    751
    58
    752
    58
    753
    58
    754
    58
    755
    58
    756
    58
    757
    58
    758
    58
    759
    58
    760
    58
    761
    59
    762
    59
    763
    59
    764
    59
    765
    60
    766
    60
    767
    60
    768
    60
    769
    60
    770
    60
    771
    60
    772
    60
    773
    60
    774
    60
    775
    60
    776
    60
    777
    60
    778
    60
    779
    61
    780
    61
    781
    61
    782
    61
    783
    61
    784
    62
    785
    62
    786
    62
    787
    62
    788
    62
    789
    62
    790
    62
    791
    63
    792
    63
    793
    63
    794
    63
    795
    64
    796
    64
    797
    64
    798
    65
    799
    66
    800
    66
    801
    66
    802
    67
    803
    68
    804
    68
    805
    68
    806
    68
    807
    68
    808
    69
    809
    69
    810
    69
    811
    69
    812
    69
    813
    69
    814
    69
    815
    69
    816
    69
    817
    69
    818
    69
    819
    69
    820
    69
    821
    69
    822
    69
    823
    69
    824
    70
    825
    71
    826
    71
    827
    71
    828
    71
    829
    71
    830
    71
    831
    71
    832
    71
    833
    71
    834
    71
    835
    71
    836
    71
    837
    71
    838
    71
    839
    72
    840
    72
    841
    72
    842
    72
    843
    72
    844
    72
    845
    72
    846
    72
    847
    72
    848
    72
    849
    73
    850
    73
    851
    73
    852
    73
    853
    74
    854
    74
    855
    74
    856
    74
    857
    74
    858
    74
    859
    74
    860
    75
    861
    76
    862
    76
    863
    76
    864
    76
    865
    76
    866
    76
    867
    77
    868
    77
    869
    77
    870
    77
    871
    77
    872
    77
    873
    77
    874
    77
    875
    77
    876
    77
    877
    77
    878
    77
    879
    78
    880
    78
    881
    78
    882
    78
    883
    78
    884
    78
    885
    78
    886
    78
    887
    79
    888
    79
    889
    79
    890
    79
    891
    80
    892
    80
    893
    80
    894
    80
    895
    80
    896
    80
    897
    80
    898
    80
    899
    80
    900
    80
    901
    80
    902
    81
    903
    82
    904
    82
    905
    82
    906
    82
    907
    82
    908
    82
    909
    82
    910
    82
    911
    82
    912
    82
    913
    82
    914
    82
    915
    82
    916
    82
    917
    82
    918
    82
    919
    82
    920
    82
    921
    83
    922
    83
    923
    84
    924
    84
    925
    84
    926
    84
    927
    85
    928
    85
    929
    85
    930
    86
    35995
    0
    0
    1
    1
    2
    1
    3
    1
    4
    2
    5
    3
    6
    3
    7
    4
    8
    4
    9
    4
    10
    4
    11
    4
    12
    4
    13
    4
    14
    5
    15
    5
    16
    5
    17
    6
    18
    7
    19
    7
    20
    8
    21
    8
    22
    8
    23
    8
    24
    8
    25
    8
    26
    8
    27
    8
    28
    8
    29
    8
    30
    8
    31
    9
    32
    10
    33
    10
    34
    10
    35
    10
    36
    10
    37
    10
    38
    10
    39
    10
    40
    10
    41
    10
    42
    10
    43
    10
    44
    10
    45
    10
    46
    10
    47
    10
    48
    10
    49
    10
    50
    10
    51
    10
    52
    10
    53
    10
    54
    10
    55
    10
    56
    11
    57
    11
    58
    12
    59
    13
    60
    14
    61
    14
    62
    14
    63
    15
    64
    15
    65
    15
    66
    15
    67
    16
    68
    16
    69
    16
    70
    16
    71
    16
    72
    16
    73
    16
    74
    16
    75
    16
    76
    16
    77
    16
    78
    16
    79
    16
    80
    16
    81
    16
    82
    16
    83
    16
    84
    16
    85
    16
    86
    16
    87
    16
    88
    16
    89
    16
    90
    16
    91
    16
    92
    16
    93
    16
    94
    16
    95
    16
    96
    16
    97
    16
    98
    17
    99
    17
    100
    17
    101
    17
    102
    17
    103
    17
    104
    17
    105
    17
    106
    17
    107
    17
    108
    17
    109
    17
    110
    17
    111
    17
    112
    17
    113
    17
    114
    17
    115
    17
    116
    17
    117
    18
    118
    18
    119
    19
    120
    19
    121
    19
    122
    19
    123
    19
    124
    19
    125
    19
    126
    19
    127
    19
    128
    19
    129
    19
    130
    19
    131
    19
    132
    19
    133
    19
    134
    19
    135
    19
    136
    19
    137
    19
    138
    19
    139
    19
    140
    19
    141
    19
    142
    19
    143
    19
    144
    19
    145
    19
    146
    19
    147
    19
    148
    19
    149
    19
    150
    19
    151
    19
    152
    19
    153
    19
    154
    19
    155
    19
    156
    19
    157
    19
    158
    19
    159
    19
    160
    19
    161
    19
    162
    19
    163
    19
    164
    19
    165
    19
    166
    19
    167
    19
    168
    19
    169
    19
    170
    19
    171
    19
    172
    19
    173
    19
    174
    19
    175
    19
    176
    19
    177
    19
    178
    19
    179
    19
    180
    19
    181
    19
    182
    20
    183
    20
    184
    20
    185
    20
    186
    21
    187
    21
    188
    22
    189
    22
    190
    22
    191
    22
    192
    22
    193
    22
    194
    22
    195
    22
    196
    22
    197
    22
    198
    22
    199
    22
    200
    22
    201
    22
    202
    22
    203
    22
    204
    22
    205
    22
    206
    22
    207
    22
    208
    22
    209
    22
    210
    22
    211
    22
    212
    22
    213
    22
    214
    22
    215
    22
    216
    22
    217
    22
    218
    22
    219
    22
    220
    22
    221
    22
    222
    22
    223
    22
    224
    23
    225
    23
    226
    23
    227
    23
    228
    23
    229
    23
    230
    23
    231
    23
    232
    23
    233
    23
    234
    23
    235
    23
    236
    23
    237
    23
    238
    23
    239
    23
    240
    23
    241
    23
    242
    23
    243
    23
    244
    23
    245
    23
    246
    23
    247
    23
    248
    23
    249
    23
    250
    23
    251
    23
    252
    23
    253
    23
    254
    23
    255
    23
    256
    23
    257
    23
    258
    23
    259
    23
    260
    23
    261
    23
    262
    23
    263
    23
    264
    23
    265
    23
    266
    23
    267
    23
    268
    23
    269
    23
    270
    23
    271
    23
    272
    23
    273
    23
    274
    23
    275
    23
    276
    23
    277
    23
    278
    23
    279
    23
    280
    23
    281
    23
    282
    23
    283
    23
    284
    23
    285
    23
    286
    23
    287
    23
    288
    23
    289
    23
    290
    23
    291
    23
    292
    23
    293
    23
    294
    23
    295
    23
    296
    23
    297
    23
    298
    23
    299
    23
    300
    23
    301
    23
    302
    23
    303
    23
    304
    23
    305
    23
    306
    23
    307
    23
    308
    23
    309
    23
    310
    23
    311
    23
    312
    23
    313
    23
    314
    23
    315
    23
    316
    24
    317
    24
    318
    25
    319
    26
    320
    27
    321
    27
    322
    27
    323
    27
    324
    28
    325
    28
    326
    29
    327
    30
    328
    30
    329
    30
    330
    31
    331
    32
    332
    32
    333
    32
    334
    32
    335
    33
    336
    33
    337
    33
    338
    33
    339
    34
    340
    35
    341
    36
    342
    37
    343
    37
    344
    37
    345
    37
    346
    38
    347
    38
    348
    38
    349
    38
    350
    38
    351
    38
    352
    39
    353
    39
    354
    40
    355
    41
    356
    41
    357
    41
    358
    41
    359
    41
    360
    41
    361
    41
    362
    41
    363
    41
    364
    41
    365
    42
    366
    43
    367
    43
    368
    43
    369
    43
    370
    43
    371
    43
    372
    43
    373
    43
    374
    43
    375
    43
    376
    43
    377
    43
    378
    43
    379
    43
:::

::: {.output .stream .stdout}
    380
    43
    381
    43
    382
    43
    383
    43
    384
    43
    385
    43
    386
    44
    387
    45
    388
    45
    389
    45
    390
    45
    391
    45
    392
    46
    393
    47
    394
    48
    395
    48
    396
    48
    397
    48
    398
    48
    399
    48
    400
    48
    401
    48
    402
    48
    403
    48
    404
    48
    405
    48
    406
    49
    407
    49
    408
    49
    409
    50
    410
    51
    411
    52
    412
    53
    413
    53
    414
    53
    415
    53
    416
    53
    417
    53
    418
    53
    419
    53
    420
    53
    421
    53
    422
    53
    423
    53
    424
    54
    425
    54
    426
    54
    427
    54
    428
    55
    429
    55
    430
    55
    431
    55
    432
    55
    433
    56
    434
    56
    435
    56
    436
    56
    437
    56
    438
    56
    439
    56
    440
    56
    441
    56
    442
    56
    443
    56
    444
    56
    445
    57
    446
    58
    447
    59
    448
    60
    449
    60
    450
    60
    451
    60
    452
    60
    453
    60
    454
    60
    455
    60
    456
    60
    457
    61
    458
    62
    459
    63
    460
    63
    461
    63
    462
    64
    463
    64
    464
    65
    465
    65
    466
    66
    467
    67
    468
    67
    469
    68
:::
:::

::: {.cell .markdown}
Then, we take the information that we have in the txt files to create
the wav files, one per vocalization.
:::

::: {.cell .code execution_count="14"}
``` python
wav_creation(fd=os.path.join(os.getcwd(), "data/longform_recordings"), baby=False)
```

::: {.output .stream .stdout}
    0
    1
:::
:::

::: {.cell .markdown}
We create a dataframe in which we have more information for each
vocalization the model detected (the day, the hour, the duration, the
probability of each label).
:::

::: {.cell .code}
``` python
df = df_pred(meta_longform, y)
```
:::

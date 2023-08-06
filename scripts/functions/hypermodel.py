import tensorflow as tf
import kerastuner as kt
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class WinWav(kt.HyperModel):

    def __init__(self, input_shape, n_labels):
        self.input_shape = input_shape
        self.n_labels = n_labels

    def build(self, hp):
        # Dimension of the input
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.BatchNormalization()(inputs)

        # Distribution in which we sample the starting coefficients
        initializer = tf.keras.initializers.HeNormal()

        # Construction of the front-end

        # Specification of the first convolutive layer, taking the input.
        # Normalization of the batch after each layer
        x = tf.keras.layers.ZeroPadding1D(padding=8896)(x)
        x = tf.keras.layers.Conv1D(filters=hp.Int('n_filters', min_value=3, max_value=24, default=12),
                                   kernel_size=hp.Int('dim_kernel', min_value=180, max_value=18000, default=980),
                                   strides=round((1 - (80 / hp.get('dim_kernel'))) * 100),
                                   padding='valid',
                                   kernel_initializer=initializer,
                                   bias_initializer=tf.keras.initializers.Constant(0.1))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=hp.Float('p_dropout', min_value=0, max_value=0.3, step=0.1, default=0.2))(x)

        # Loop to increment the layers
        for i in range(hp.Int('n_layers', min_value=1, max_value=4, default=3)):
            x = tf.keras.layers.ZeroPadding1D(padding=8896)(x)
            x = tf.keras.layers.Conv1D(filters=hp.Int('n_filters_' + str(i), min_value=3, max_value=24, default=12),
                                       kernel_size=hp.Int('dim_kernel_' + str(i), min_value=180, max_value=18000, default=980),
                                       strides=round((1 - (80 / hp.get('dim_kernel_' + str(i)))) * 100),
                                       padding='valid',
                                       kernel_initializer=initializer,
                                       bias_initializer=tf.keras.initializers.Constant(0.1))(x)
            x = tf.keras.layers.PReLU()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(rate=hp.Float('p_dropout', min_value=0, max_value=0.3, step=0.1, default=0.1))(x)

        # Global mean of the data (i.e., for each dimension of the kernel)
        representation = tf.keras.layers.GlobalMaxPool1D(name='representation')(x)

        # Construction of two modules from the latent space.
        # One to distinguish between the different vocalizations of the species
        mod_multi = tf.keras.layers.Dense(units=hp.Int('n_units', min_value=32, max_value=384, default=256),
                                          kernel_initializer=initializer,
                                          bias_initializer=tf.keras.initializers.Constant(0.1),
                                          kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(representation)
        mod_multi = tf.keras.layers.PReLU()(mod_multi)
        mod_multi = tf.keras.layers.BatchNormalization()(mod_multi)
        mod_multi = tf.keras.layers.Dropout(rate=hp.Float('p_dropout_fc', min_value=0, max_value=0.9, step=0.1, default=0.5))(mod_multi)

        # Loop to increment layers
        for i in range(hp.Int('n_fc_layers', min_value=0, max_value=2, default=0)):
            mod_multi = tf.keras.layers.Dense(units=hp.Int('n_units_' + str(i), min_value=32, max_value=384, default=256),
                                              kernel_initializer=initializer,
                                              bias_initializer=tf.keras.initializers.Constant(0.1),
                                              kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(mod_multi)
            mod_multi = tf.keras.layers.PReLU()(mod_multi)
            mod_multi = tf.keras.layers.BatchNormalization()(mod_multi)
            mod_multi = tf.keras.layers.Dropout(rate=hp.Float('p_dropout_fc', min_value=0, max_value=0.9, step=0.1, default=0.5))(mod_multi)

        y_multi = tf.keras.layers.Dense(self.n_labels, activation='softmax', name='output_multi')(mod_multi)

        # Another to distinguish between a vocalization and noise
        mod_binaire = tf.keras.layers.Dense(units=hp.Int('n_units', min_value=32, max_value=384, default=256),
                                            kernel_initializer=initializer,
                                            bias_initializer=tf.keras.initializers.Constant(0.1),
                                            kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(representation)
        mod_binaire = tf.keras.layers.PReLU()(mod_binaire)
        mod_binaire = tf.keras.layers.BatchNormalization()(mod_binaire)
        mod_binaire = tf.keras.layers.Dropout(rate=hp.Float('p_dropout_fc', min_value=0, max_value=0.9, step=0.1, default=0.5))(mod_binaire)

        # Loop to increment layers
        for i in range(hp.Int('n_fc_layers', min_value=0, max_value=2, default=0)):
            mod_binaire = tf.keras.layers.Dense(units=hp.Int('n_units_' + str(i), min_value=32, max_value=384, default=256),
                                                kernel_initializer=initializer,
                                                bias_initializer=tf.keras.initializers.Constant(0.1),
                                                kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(mod_binaire)
            mod_binaire = tf.keras.layers.PReLU()(mod_binaire)
            mod_binaire = tf.keras.layers.BatchNormalization()(mod_binaire)
            mod_binaire = tf.keras.layers.Dropout(rate=hp.Float('p_dropout_fc', min_value=0, max_value=0.9, step=0.1, default=0.5))(mod_binaire)

        y_binaire = tf.keras.layers.Dense(1, activation='sigmoid', name='output_binaire')(mod_binaire)

        # Creation of the model, with inputs and outputs
        model = tf.keras.Model(inputs=inputs, outputs=[y_binaire, y_multi], name='win_wav')

        # Algorithm Nadam to compute the gradient, Adam (Kingma et Ba, 2017) +
        # Nestertov momentum
        algorithme = tf.keras.optimizers.Nadam(learning_rate=hp.Float('learning_rate', min_value=1e-10, max_value=1e-02, default=1e-06),
                                              beta_1=hp.Float('rho1', min_value=0, max_value=0.9, default=0.9),
                                              beta_2=hp.Float('rho2', min_value=0.99, max_value=0.9999, default=0.999))
        # Metric to compute
        metrics_multi = [tf.keras.metrics.categorical_accuracy]
        metrics_binaire = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve='PR'),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

        # Compile the model, with the losses for each module
        model.compile(optimizer=algorithme,
                      loss={'output_multi': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                            'output_binaire': tf.keras.losses.BinaryCrossentropy(from_logits=False)},
                      loss_weights={'output_multi': 1.0, 'output_binaire': 1.0},
                      metrics={'output_multi': metrics_multi,
                               'output_binaire': metrics_binaire})

        return model


class WinWavTransferLearning(kt.HyperModel):

    def __init__(self, input_shape, n_labels, two_loss=True):
        self.input_shape = input_shape
        self.n_labels = n_labels
        self.two_loss = two_loss

    def build(self, hp):
        # Representation from YamNet
        input_representation = tf.keras.layers.Input(shape=self.input_shape, dtype=tf.float32, name='input_classification')
        x = tf.keras.layers.BatchNormalization()(input_representation)

        # Module to predict the label of the vocalization, from the embedding
        for i in range(hp.Int('n_layers_classification', min_value=1, max_value=6, default=2)):
            mod_multi = tf.keras.layers.Dense(units=hp.Int('n_units_classification_' + str(i), min_value=32, max_value=1024, default=512),
                                              kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer=tf.keras.initializers.Constant(0.1),
                                              kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(x)
            mod_multi = tf.keras.layers.PReLU()(mod_multi)
            mod_multi = tf.keras.layers.BatchNormalization()(mod_multi)
            mod_multi = tf.keras.layers.Dropout(hp.Float('p_droupout', min_value=0, max_value=0.9, step=0.1, default=0.5))(mod_multi)

        y_multi = tf.keras.layers.Dense(self.n_labels, activation='softmax', name='output_multi')(mod_multi)

        # Module to distinguish between noise and vocalization, from the embedding
        for i in range(hp.Int('n_layers_detection', min_value=1, max_value=6, default=2)):
            mod_binaire = tf.keras.layers.Dense(units=hp.Int('n_units_detection_' + str(i), min_value=32, max_value=1024, default=512),
                                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                                bias_initializer=tf.keras.initializers.Constant(0.1),
                                                kernel_constraint=tf.keras.constraints.max_norm(hp.Int('max_norm', min_value=0, max_value=8, step=1, default=4)))(x)
            mod_binaire = tf.keras.layers.PReLU()(mod_binaire)
            mod_binaire = tf.keras.layers.BatchNormalization()(mod_binaire)
            mod_binaire = tf.keras.layers.Dropout(hp.Float('p_droupout', min_value=0.1, max_value=0.9, step=0.1, default=0.5))(mod_binaire)

        y_binaire = tf.keras.layers.Dense(1, activation='sigmoid', name='output_binaire')(mod_binaire)

        # Learning Algorithm
        algorithme = tf.keras.optimizers.Nadam(learning_rate=hp.Float('learning_rate', min_value=1e-10, max_value=1e-02, default=1e-06),
                                               beta_1=hp.Float('rho1', min_value=0, max_value=0.9, default=0.9),
                                               beta_2=hp.Float('rho2', min_value=0.99, max_value=0.9999, default=0.999))

        # Metrics
        metrics_multi = [tf.keras.metrics.categorical_accuracy]
        metrics_binaire = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve='PR'),
                           tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
                           tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                           tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

        if self.two_loss is True:
            # Creation of the model
            model = tf.keras.Model(inputs=input_representation, outputs=[y_binaire, y_multi], name='model_classi')

            # Specification of the loss functions for each module
            model.compile(optimizer=algorithme,
                          loss={'output_multi': tf.keras.losses.CategoricalCrossentropy(),
                                'output_binaire': tf.keras.losses.BinaryCrossentropy()},
                          loss_weights={'output_multi': 1.0, 'output_binaire': 1.0},
                          metrics={'output_multi': metrics_multi,
                                   'output_binaire': metrics_binaire})
        else:
            # if no information about the label of the vocalization, construction
            # of a model for the detection only
            model = tf.keras.Model(inputs=input_representation, outputs=y_binaire, name='model_classi')
            model.compile(optimizer=algorithme,
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=metrics_binaire)

        return model

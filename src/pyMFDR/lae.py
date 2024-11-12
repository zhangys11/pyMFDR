from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler

import warnings
warnings.warn("deprecated", DeprecationWarning)

class LAE:

    def __init__(self, n_components = 64):
        """
        Initialization / Constructor
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.ae = None
        self.encoder = None
        self.decoder = None
        self.hist = None
        self.components_ = None
        
        
    def fit(self, X, epochs = 200, batch_size = 8, l1_reg = 0.01, l2_reg = 0.01, verbose = 0):
    
        self.X_scaled = self.scaler.fit_transform(X)  
        self.encoder, self.decoder, self.ae, self.hist = build_1_linear_dense_layer_auto_encoder(
        self.X_scaled, encoding_dim = self.n_components, 
        epochs = epochs, batch_size = batch_size, 
        l1_reg = l1_reg, l2_reg = l2_reg, verbose = verbose)
        
        self.components_ = self.ae.layers[2].get_weights()[0]
        
        
    def fit_transform(self, X, epochs = 200, batch_size = 8, l1_reg = 0.01, l2_reg = 0.01, verbose = 0):
    
        self.fit(X, epochs = epochs, batch_size = batch_size, 
                 l1_reg = l1_reg, l2_reg = l2_reg, verbose = verbose)
        
        Z = self.encoder.predict(self.X_scaled)
        
        return Z
        
    def inverse_transform(self, Z):
    
        Xr_scaled = self.decoder.predict(Z)
        Xr = self.scaler.inverse_transform(Xr_scaled)
        
        return Xr        
           
        
def build_1_linear_dense_layer_auto_encoder(X, encoding_dim = 64, 
                                        epochs = 200, batch_size = 8, 
                                        l1_reg = 0.01, l2_reg = 0.01, verbose = 0):    
    """
    Define and compile an auto encoder for dimension reduction purposes.
    AutoEncoder (as well as other NN models, such as MLP) is sensitive to feature scaling, so it is highly recommended to scale your data.
    encoding_dim: the size of our encoded representations. Default value is 64. For a general Raman spectroscoopic data, this is about 3%(64/2090) compression ratio        
    """

    original_dim = X.shape[1]


    # this is our input placeholder
    input_layer = Input(shape=(original_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, 
                    activation=None, # or 'relu'. # activation: If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
                   )(input_layer)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(original_dim, 
                    activation=None, # or 'sigmoid'. 
                    kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                   )(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_layer, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=4,
                       verbose=0, 
                       mode='auto')

    ckp = ModelCheckpoint(filepath="temp_weights_checkpoint.hdf5", 
                      verbose=1, 
                      save_best_only=True)

    hist = autoencoder.fit(X, X,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, # shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch')
                    validation_split = 0.2,
                    callbacks = [es], # , ckp
                   verbose = verbose) # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    return encoder, decoder, autoencoder, hist
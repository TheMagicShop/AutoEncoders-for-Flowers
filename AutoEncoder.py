from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, Flatten, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf

class AutoEncoder():
    def __init__(self, input_shape, layer_filters,
                 activation, latent_dim):
        self.input_shape = input_shape
        self.layer_filters = layer_filters
        self.kernel_size = 3
        self.activation = activation
        self.latent_dim = latent_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()
    
    def _build_encoder(self):
        inputs = Input(self.input_shape, name='encoder_input')
        x = inputs
        for filters in self.layer_filters:
            x = Conv2D(filters,
                       kernel_size=self.kernel_size,
                       activation=self.activation,
                       strides=2,
                       padding='same')(x)
        self.shape = x.shape[1:]
        x = Flatten()(x)
        latent = Dense(self.latent_dim, name='latent_vector')(x)
        
        model = Model(inputs, latent, name='Encoder')
        return model
    
    def _build_decoder(self):
        inputs = Input(self.latent_dim, name='decoder_input')
        x = Dense(self.shape[0] * self.shape[1] * self.shape[2])(inputs) 
        x = Reshape(self.shape)(x)
        for filters in self.layer_filters[::-1]:
            x = Conv2DTranspose(filters,
                                kernel_size=self.kernel_size,
                                activation=self.activation,
                                strides=2,
                                padding='same')(x)
        outputs = Conv2D(self.input_shape[-1],
                         kernel_size=1,
                         activation='sigmoid',
                         padding='same',
                         name='decoder_output')(x)
        
        model = Model(inputs, outputs, name='Decoder')
        return model
    
    def _build_autoencoder(self):
        inputs = Input(shape=self.input_shape, name='autoencoder_input')
        outputs = self.decoder(self.encoder(inputs))
        model = Model(inputs, outputs, name='AutoEncoder')
        return model

    def train(self, 
              train_gen,  
              val_gen=None, 
              epochs=1, 
              steps_per_epoch=1,
              optimizer='adam',
              loss='mse', 
              vloss_weight=0.5,
              callbacks=None):
                    
        self.autoencoder.compile(optimizer=optimizer,
                                 loss=[loss, self._total_variation_loss],
                                 loss_weights=[1 - vloss_weight, vloss_weight],
                                 )
        self.autoencoder.fit(x=train_gen,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=val_gen,
                             callbacks=callbacks
                             )           
        
    @tf.function
    def _total_variation_loss(X_in, X_out):
        vertical_variation = tf.square(
            X_out[:, 1:, :, :] - X_out[:, :-1, :, :])
        horizontal_variation = tf.square(
            X_out[:, :, 1:, :] - X_out[:, :, :-1, :])
        total_variation_loss = tf.sqrt(tf.reduce_sum(
                        horizontal_variation + vertical_variation))
        return total_variation_loss        
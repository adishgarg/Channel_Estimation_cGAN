import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt




config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The architecture of generator is a modified U-Net.
There are skip connections between the encoder and decoder (as in U-Net).
"""


class EncoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s = 2, apply_batchnorm=True, add = False, padding_s = 'same'):
        super(EncoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides_s,
                             padding=padding_s, kernel_initializer=initializer, use_bias=False)
        ac = layers.LeakyReLU()
        self.encoder_layer = None
        if add:
            self.encoder_layer = tf.keras.Sequential([conv])
        elif apply_batchnorm:
            bn = layers.BatchNormalization()
            self.encoder_layer = tf.keras.Sequential([conv, bn, ac])
        else:
            self.encoder_layer = tf.keras.Sequential([conv, ac])

    def call(self, x):
        return self.encoder_layer(x)


class DecoderLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides_s = 2, apply_dropout=False, add = False):
        super(DecoderLayer, self).__init__()
        initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
        dconv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides_s,
                                       padding='same', kernel_initializer=initializer, use_bias=False)
        bn = layers.BatchNormalization()
        ac = layers.ReLU()
        self.decoder_layer = None
        
        if add:
            self.decoder_layer = tf.keras.Sequential([dconv])      
        elif apply_dropout:
            drop = layers.Dropout(rate=0.5)
            self.decoder_layer = tf.keras.Sequential([dconv, bn, drop, ac])
        else:
            self.decoder_layer = tf.keras.Sequential([dconv, bn, ac])
            
        
            

    def call(self, x):
        return self.decoder_layer(x)
    
class SelfAttention(layers.Layer):
    def __init__(self, filters):
        super(SelfAttention, self).__init__()
        self.query_conv = layers.Conv2D(filters // 8, kernel_size=1, padding="same", kernel_initializer="he_normal")
        self.key_conv = layers.Conv2D(filters // 8, kernel_size=1, padding="same", kernel_initializer="he_normal")
        self.value_conv = layers.Conv2D(filters, kernel_size=1, padding="same", kernel_initializer="he_normal")
        self.gamma = tf.Variable(initial_value=0.0, trainable=True)  # Trainable scalar
    
    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]

        # Compute query, key, and value
        query = self.query_conv(inputs)  # (B, H, W, F')
        key = self.key_conv(inputs)      # (B, H, W, F')
        value = self.value_conv(inputs)  # (B, H, W, F)

        # Reshape for compatibility in matrix multiplication
        query = tf.reshape(query, [batch_size, -1, channels // 8])  # (B, H*W, F')
        key = tf.reshape(key, [batch_size, channels // 8, -1])      # (B, F', H*W)
        value = tf.reshape(value, [batch_size, -1, channels])       # (B, H*W, F)

        # Compute attention map and output
        attention_map = tf.nn.softmax(tf.matmul(query, key), axis=-1)  # (B, H*W, H*W)
        out = tf.matmul(attention_map, value)  # (B, H*W, F)
        out = tf.reshape(out, [batch_size, height, width, channels])  # (B, H, W, C)

        # Add residual connection
        return self.gamma * out + inputs


# Generator Class
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Initial Resizing Layers (placeholders for your DecoderLayer logic)
        self.p_layers = [
            DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True),
            DecoderLayer(filters=2, kernel_size=4, strides_s=2, apply_dropout=False, add=True),
            EncoderLayer(filters=2, kernel_size=(6, 1), strides_s=(4, 1), apply_batchnorm=False, add=True)
        ]
        
        # Encoder
        self.encoder_layers = [
            EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False),
            EncoderLayer(filters=128, kernel_size=4),
            EncoderLayer(filters=256, kernel_size=4),
            EncoderLayer(filters=512, kernel_size=4),
            EncoderLayer(filters=512, kernel_size=4)
        ]

        # Self-Attention Layer after Encoder
        self.attention = SelfAttention(filters=512)

        # Decoder
        self.decoder_layers = [
            DecoderLayer(filters=512, kernel_size=4, apply_dropout=True),
            DecoderLayer(filters=512, kernel_size=4, apply_dropout=True),
            DecoderLayer(filters=256, kernel_size=4, apply_dropout=False),
            DecoderLayer(filters=128, kernel_size=4, apply_dropout=False)
        ]

        # Output Layer
        self.last = layers.Conv2DTranspose(
            filters=2, kernel_size=4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
            activation='tanh'
        )

    def call(self, x):
        # Pre-processing layers
        for p_layer in self.p_layers:
            x = p_layer(x)

        # Encoder
        encoder_xs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_xs.append(x)
        
        # Apply Self-Attention
        x = self.attention(x)

        # Skip connections (reverse order for decoder)
        encoder_xs = encoder_xs[:-1][::-1]
        assert len(encoder_xs) == 4  # Ensure matching skip connections

        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(x)
            x = tf.concat([x, encoder_xs[i]], axis=-1)  # Skip connections

        # Final layer
        return self.last(x)


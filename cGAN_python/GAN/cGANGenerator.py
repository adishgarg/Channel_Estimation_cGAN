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
    
class EnhancedSelfAttention(layers.Layer):
    def __init__(self, filters, pilot_length, num_heads=4):
        super(EnhancedSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.filters = filters
        self.pilot_length = pilot_length  # Length of pilots corresponds to temporal relations
        self.head_dim = filters // num_heads
        
        assert filters % num_heads == 0, "Filters must be divisible by the number of heads."

        # Attention layers
        self.query_conv = layers.Conv2D(filters, kernel_size=1, padding="same")
        self.key_conv = layers.Conv2D(filters, kernel_size=1, padding="same")
        self.value_conv = layers.Conv2D(filters, kernel_size=1, padding="same")
        self.output_conv = layers.Conv2D(filters, kernel_size=1, padding="same")

        # Normalization layers
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()

        # Residual scaling factor
        self.gamma = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)

    def split_heads(self, x, batch_size):
        # Splits channels into heads and reshapes for multi-head attention
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.head_dim])  # Flatten pilot dims
        return tf.transpose(x, [0, 2, 1, 3])  # (B, num_heads, flattened_dim, head_dim)

    def merge_heads(self, x, batch_size, height, width):
        # Merges heads back into the original feature dimension
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, flattened_dim, num_heads, head_dim)
        x = tf.reshape(x, [batch_size, height, width, self.filters])  # (B, H, W, C)
        return x

    def call(self, inputs):
        # Input shape: (B, bs_ant, pilot_length, 2)
        batch_size = tf.shape(inputs)[0]
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]

        # Normalize input data
        x = self.layer_norm1(inputs)

        # Compute query, key, value
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention
        attention_logits = tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(self.head_dim))
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)  # (B, num_heads, flattened_dim, flattened_dim)

        # Apply attention weights to values
        attention_output = tf.matmul(attention_weights, value)  # (B, num_heads, flattened_dim, head_dim)

        # Merge heads
        attention_output = self.merge_heads(attention_output, batch_size, height, width)

        # Output projection and residual connection
        attention_output = self.output_conv(attention_output)
        attention_output = self.gamma * attention_output + inputs  # Residual connection

        # Normalize output
        return self.layer_norm2(attention_output)



# Generator Class
c# Updated Generator Class
c# Updated Generator Class
class Generator(tf.keras.Model):
    def __init__(self, pilot_length=8):  # Include pilot_length for attention
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

        # Updated Self-Attention Layer after Encoder
        self.attention = EnhancedSelfAttention(filters=512, pilot_length=pilot_length, num_heads=4)

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
        
        # Apply Enhanced Self-Attention
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


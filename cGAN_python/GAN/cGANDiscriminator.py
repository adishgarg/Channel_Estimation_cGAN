import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from GAN.cGANGenerator import EncoderLayer
import os


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers

"""
The Discriminator is a PatchGAN.
"""
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


class Discriminator(tf.keras.Model):
    def __init__(self, pilot_length=8):
        """
        Discriminator for a cGAN with EnhancedSelfAttention.
        Uses an encoder-like downsampling structure with attention and final logits.
        """
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        # Encoder Layers (Downsampling)
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False) 
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)        
        self.encoder_layer_3 = EncoderLayer(filters=256, kernel_size=4)

        # Enhanced Self-Attention Block
        self.attention = EnhancedSelfAttention(filters=256, pilot_length=pilot_length, num_heads=4)

        # Intermediate Block
        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv = layers.Conv2D(512, 4, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.ac = layers.LeakyReLU()

        # Output Block
        self.zero_pad2 = layers.ZeroPadding2D()
        self.last = layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=initializer)

    def call(self, inputs):
        """
        Forward pass for the Discriminator model.
        Inputs:
            inputs: Tensor of shape (batch_size, bs_ant, pilot_length, 2)
        Returns:
            Tensor of shape (batch_size, H, W, 1) representing realness logits.
        """
        x = inputs

        # Pass through Encoder Layers
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        # Apply Enhanced Self-Attention
        x = self.attention(x)

        # Intermediate Convolution Block
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        # Final Output Block
        x = self.zero_pad2(x)
        return self.last(x)

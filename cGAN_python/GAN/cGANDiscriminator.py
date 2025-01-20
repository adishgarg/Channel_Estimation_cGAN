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


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        # Encoder Layers (Downsampling)
        self.encoder_layer_1 = EncoderLayer(filters=64, kernel_size=4, apply_batchnorm=False) 
        self.encoder_layer_2 = EncoderLayer(filters=128, kernel_size=4)        
        self.encoder_layer_3 = EncoderLayer(filters=256, kernel_size=4)

        # Self-Attention Block
        self.attention = SelfAttention(filters=256)

        # Intermediate Block
        self.zero_pad1 = layers.ZeroPadding2D()
        self.conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.ac = layers.LeakyReLU()

        # Output Block
        self.zero_pad2 = layers.ZeroPadding2D()
        self.last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)

    def call(self, inputs):
        """Pass inputs (e.g., real or generated images) through the model."""
        x = inputs
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        x = self.encoder_layer_3(x)

        # Self-Attention Integration
        x = self.attention(x)

        # Intermediate Convolutions
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.ac(x)

        # Final Discriminator Logit
        x = self.zero_pad2(x)
        return self.last(x)












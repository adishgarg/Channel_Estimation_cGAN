import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from GAN.cGANGenerator import Generator
from GAN.cGANDiscriminator import Discriminator
from GAN.data_preprocess import load_image_train, load_image_test, load_image_test_y
from tempfile import TemporaryFile
from scipy.io import loadmat, savemat
import datetime
import h5py
import hdf5storage
import skfuzzy as fuzz
import pandas as pd
import matplotlib
# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
matplotlib.use('Agg')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
layers = tf.keras.layers



# data path
path = "Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat"


# batch = 1 produces good results on U-NET
BATCH_SIZE = 1              

# model
generator = Generator()
discriminator = Discriminator()
# optimizer
generator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_optimizer = tf.compat.v1.train.RMSPropOptimizer(2e-5)
#discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(2e-4, beta1=0.5)

"""
Discriminator loss:
The discriminator loss function takes 2 inputs; real images, generated images
real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since the real images)
generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since the fake images)
Then the total_loss is the sum of real_loss and the generated_loss

Generator loss:
It is a sigmoid cross entropy loss of the generated images and an array of ones.
The paper also includes L2 loss between the generated image and the target image.
This allows the generated image to become structurally similar to the target image.
The formula to calculate the total generator loss = gan_loss + LAMBDA * l2_loss, where LAMBDA = 100. 
This value was decided by the authors of the paper.
"""


def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), logits=disc_real_output)  # label=1
    generated_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_generated_output), logits=disc_generated_output)  # label=0
    total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
    # L2 loss
    l2_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = tf.reduce_mean(gen_loss) + l2_weight * l2_loss
    return total_gen_loss

def generated_image(model, test_input, tar, t=0):
    """Display and save the results of Generator, and save data to Excel."""    
    prediction = model(test_input)

    # Squeeze the data to remove unnecessary dimensions
    input_data = np.squeeze(test_input[:,:,:,0])
    target_data = np.squeeze(tar[:,:,:,0])
    prediction_data = np.squeeze(prediction[:,:,:,0])

    # Save the data in Excel
    data = {
        'Input Y': input_data.flatten(),
        'Target H': target_data.flatten(),
        'Prediction H': prediction_data.flatten(),
    }
    df = pd.DataFrame(data)

    # Ensure the directory exists
    os.makedirs("cGAN_python/generated_img", exist_ok=True)

    # Save the DataFrame to Excel
    excel_filename = os.path.join("cGAN_python/generated_img", f"data_{t}.xlsx")
    df.to_excel(excel_filename, index=False)
    print(f"Excel file saved as: {excel_filename}")

    # Save the plot for debugging or visualization purposes
    try:
        plt.figure()
        plt.plot(target_data.flatten(), label="Target")
        plt.plot(prediction_data.flatten(), label="Prediction")
        plt.title('Target vs Prediction')
        plt.legend()
        plot_filename = os.path.join("cGAN_python/generated_img", f"comparison_{t}.png")
        plt.savefig(plot_filename)
        print(f"Comparison plot saved as: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Error generating or saving the plot: {e}")


def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image)                      # input -> generated_target
        disc_real_output = discriminator(target)  # [input, target] -> disc output
        disc_generated_output = discriminator(gen_output)  # [input, generated_target] -> disc output
        # print("*", gen_output.shape, disc_real_output.shape, disc_generated_output.shape)

        # calculate loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target)   # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss

    # gradient
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # apply gradient
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(epochs):
    nm = []
    ep = []
    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        print("-----\nEPOCH:", epoch)
        # Train
        for bi, (target, input_image) in enumerate(load_image_train(path)):
            elapsed_time = datetime.datetime.now() - start_time
            gen_loss, disc_loss = train_step(input_image, target)
            print(f"B/E: {bi}/{epoch}, Generator loss: {gen_loss.numpy()}, Discriminator loss: {disc_loss.numpy()}, Time: {elapsed_time}")

        # Calculate NMSE for the epoch
        realim, inpuim = load_image_test_y(path)
        prediction = generator(inpuim)
        nmse_value = fuzz.nmse(np.squeeze(realim), np.squeeze(prediction))
        nm.append(nmse_value)
        ep.append(epoch + 1)

        # Save NMSE data to an Excel file at each epoch
        df = pd.DataFrame({"Epoch": ep, "NMSE": nm})
        df.to_excel("nmse_epoch.xlsx", index=False)

    # Plot and save NMSE graph after training
    plt.figure()
    plt.plot(ep, nm, '-r', label="NMSE")
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('NMSE Across Epochs')
    plt.legend()
    plt.savefig("nmse_graph.png")
    plt.close()

    return nm, ep

if __name__ == "__main__":

    # train
    nm, ep = train(epochs=10)
    
    plt.figure()
    plt.plot(ep,nm,'^-r')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.savefig("output.png")

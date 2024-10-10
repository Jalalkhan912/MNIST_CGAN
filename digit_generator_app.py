import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

# Load your generator model here (assuming it's saved and ready)
loaded_generator = tf.keras.models.load_model('generator.h5')
latent_dim = 100  # Set this to your latent dimension

# Function to generate specific digits
def generate_specific_digits(loaded_generator, digit, num_samples=5):
    noise = tf.random.normal([num_samples, latent_dim])
    labels = tf.fill([num_samples, 1], digit)
    generated_images = loaded_generator([noise, labels], training=False)
    generated_images = generated_images * 0.5 + 0.5  # Rescale to [0, 1]
    return generated_images

# Function to plot generated digits
def plot_generated_digits(images, digit):
    num_images = images.shape[0]
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axs[i].imshow(images[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Digit: {digit}')
    plt.tight_layout()

    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Streamlit UI
st.title("Handwritten Digit Generator")
st.write("Generate images of specific handwritten digits using a GAN.")

# Input for the digit to generate
digit_to_generate = st.number_input("Enter a digit (0-9):", min_value=0, max_value=9, value=4)
num_samples = st.number_input("Number of samples to generate:", min_value=1, max_value=25, value=5)

if st.button("Generate"):
    with st.spinner("Generating images..."):
        generated_images = generate_specific_digits(loaded_generator, digit_to_generate, num_samples)
        image_buf = plot_generated_digits(generated_images, digit_to_generate)

    st.image(image_buf, caption=f"Generated {num_samples} images of digit {digit_to_generate}.")
    st.success("Images generated successfully!")

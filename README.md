# Simple GAN for MNIST Digit Generation

## Description
This project implements a basic Generative Adversarial Network (GAN) to generate synthetic images of handwritten digits. It is trained on the classic MNIST dataset.

A GAN consists of two neural networks:
-   **Generator:** Takes random noise as input and attempts to create a realistic image.
-   **Discriminator:** Takes an image as input and attempts to classify it as either "real" (from the dataset) or "fake" (from the generator).

The two networks are trained in a zero-sum game where the generator gets better at fooling the discriminator, and the discriminator gets better at catching the generator. Over time, the generator learns to produce highly realistic images.

## Features
-   Generator and Discriminator networks implemented in PyTorch.
-   Uses the MNIST dataset, downloaded automatically by `torchvision`.
-   Training loop that alternates between training the discriminator and the generator.
-   Saves a grid of generated images at the end of training.

## Setup and Installation

1.  **Clone the repository and navigate to the directory.**
2.  **Create a virtual environment and activate it.**
3.  **Install the dependencies:** `pip install -r requirements.txt`
4.  **Run the script:** `python src/main.py`

## Example Output
```
[Epoch 1/50] [Batch 100/938] [D loss: 0.512] [G loss: 1.154]
[Epoch 1/50] [Batch 200/938] [D loss: 0.621] [G loss: 0.987]
...
[Epoch 50/50] [Batch 900/938] [D loss: 0.675] [G loss: 0.781]
Training finished.
Generated images saved to assets/gan_generated_images.png
```
*(An image file `gan_generated_images.png` showing a grid of generated digits will be created in the `assets` folder.)*

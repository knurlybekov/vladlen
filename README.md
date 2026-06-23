# vladlen

A Deep Convolutional GAN (DCGAN) built with TensorFlow/Keras that generates 128×128 abstract artworks in the style of Soviet-era propaganda posters.

## Overview

This project trains a DCGAN on a curated set of Soviet propaganda images and learns to synthesize new 128×128 abstract pieces in that visual style. It was presented as an exploration of generative adversarial networks for stylized art generation.

## Architecture

- **Generator** — dense projection → reshape to 8×8 → stacked `Conv2DTranspose` upsampling (8→16→32→64→128) with BatchNorm + LeakyReLU, `tanh` output
- **Discriminator** — strided `Conv2D` stack with LeakyReLU + Dropout, single logit output
- **Loss** — binary cross-entropy (from logits) for both networks
- Training checkpoints saved under `training_checkpoints/`

## Tech Stack

Python · TensorFlow · Keras

## Usage

```bash
pip install tensorflow
# place training images (.jpg) in a 'propoganda/' folder
python main.py
```

Images are normalized to [-1, 1] and resized to 128×128 before training. A presentation deck describing the approach is included in the repo.

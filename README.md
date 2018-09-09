# Heated-Up Softmax Embedding
<img src="./fig/car196.png" width="500">

## Introduction

Descriptor extraction is mapping an image to a point in the descriptor space.

<img src="./fig/pipeline.png" width="500">

**A good descriptor should be:**

   * Compact: Samples from the same class to be close
   * Spread-out: Samples from different classes to be far away

Bottleneck feature from classification network shows strong performance in [1]

<img src="./fig/bottleneck.png" width="400">

Bottleneck feature is not learned to be compact and spread-out, thus may not be suitable for clustering and retreival. Here are some features learned from MNIST. Each color shows one digit. The diamond shows classifier weight. 

<img src="./fig/off-the-shelf.png" width="400">

## Temperature parameter $\alpha$


import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, key, h_dim):
        key1, key2 = jax.random.split(key, 2)
        self.conv1 = nn.Conv2d(h_dim, h_dim, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(h_dim, h_dim, kernel_size=9, padding=4)

    def __call__(self, x):
        h = nn.relu(self.conv1(x))
        y = x + self.conv2(h)
        return y

class MnistDiffusion(eqx.Module): 
    layers: list
    def __init__(self, key, h_dim):
        keys = jax.random.split(key, 9)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(2, h_dim, kernel_size=1, key=keys[0]),
            ResBlock(keys[1], h_dim),
            ResBlock(keys[2], h_dim),
            ResBlock(keys[3], h_dim),
            ResBlock(keys[4], h_dim),
            ResBlock(keys[5], h_dim),
            ResBlock(keys[6], h_dim),
            ResBlock(keys[7], h_dim),
            eqx.nn.Conv2d(h_dim, 1, kernel_size=1, key=keys[8])
        ]
    def __call__(self, x, gamma):
        x = jnp.stack([x,jnp.ones_like(x)*gamma])
        for layer in self.layers:
            x = layer(x)
        x = x[0]
        return x

class MnistClassifier(eqx.Module): 
    layers: list
    head: eqx.nn.Linear
    def __init__(self, key, h_dim):
        keys = jax.random.split(key, 9)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, h_dim, kernel_size=1, key=keys[0]),
            ResBlock(keys[1], h_dim),
            ResBlock(keys[2], h_dim),
            ResBlock(keys[3], h_dim),
            ResBlock(keys[4], h_dim),
            ResBlock(keys[5], h_dim),
            ResBlock(keys[6], h_dim),
            eqx.nn.Conv2d(h_dim, 1, kernel_size=1, key=keys[7])
        ]
        self.head = eqx.nn.Linear(28*28, 10, key=keys[8])
    def __call__(self, x):
        x = x[None,:,:]
        for layer in self.layers:
            x = layer(x)
        x = self.head(x.reshape((-1,)))
        x = jax.nn.log_softmax(x)
        return x 
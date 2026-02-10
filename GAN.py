from NN import *
import random

"""
GAN from scratch
Using the Neural network model made in NN.py file
we only need to create two models, a generator and a discriminator
and the custom training loop for the GAN
"""

generator_layers = [Dense(1,4), ReLU(), Dense(4,3), Sigmoid()]
Generator = Model(generator_layers)

discriminator_layers = [Dense(3,4), ReLU(), Dense(4,1), Sigmoid()]
Discriminator = Model(discriminator_layers)

opt_G = SGD(Generator.parameters(), lr=0.1)
opt_D = SGD(Discriminator.parameters(), lr=0.1)
loss_fn = BinaryCrossEntropy()

def train_gan(epochs):
  target_value = np.array([0.35, 0.05, 0.05])

  for epoch in range(epochs):
    """
    In the first phase of the training loop, we fix the training of the generator
    and we train the discriminator as a normal classification model given
    m real samples and m generated fake samples, with the goal of accurately differentiating
    between real and generated samples
    """
    opt_D.zero_grad()

    #we train for real data
    x_real = np.array(target_value)
    pred_real = Discriminator.forward(x_real)
    loss_real = loss_fn.forward(pred_real, [1.0])
    grad_real = loss_fn.backward()
    Discriminator.backward(x_real, grad_real)

    #we train for fake data
    z = [random.uniform(0,1)]
    x_fake = Generator.forward(z)
    pred_fake = Discriminator.forward(x_fake)
    loss_fake = loss_fn.forward(pred_fake, [0.0])
    grad_fake = loss_fn.backward()
    Discriminator.backward(x_fake, grad_fake)

    opt_D.step()

    """
    Now we train the generator by backpropagating from loss to discriminator to generator
    """
    opt_G.zero_grad()
    z = [random.uniform(0,1)]
    x_fake = Generator.forward(z)
    pred_fakeG = Discriminator.forward(x_fake)
    loss_G = loss_fn.forward(pred_fakeG, [1.0])
    grad_from_D = loss_fn.backward()
    grad_at_pixels = Discriminator.backward(x_fake, grad_from_D)
    Generator.backward(z, grad_at_pixels)
    opt_G.step()

    if epoch % 200 == 0:
      current_gen = Generator.forward([0.5])[0]
      print(f"Epoch {epoch} | Gen Output: {current_gen:.4f} | D(Real): {pred_real[0]:.4f}")


print("Starting GAN training  ")
train_gan(2000)
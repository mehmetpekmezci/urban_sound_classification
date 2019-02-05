import tensorflow as tf
import numpy as np

def compute_cost(Z_L, Y):
    cost = tf.norm(Z_L-Y) #L2 distance between guesses
    return cost
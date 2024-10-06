from typing import NamedTuple
from jax import random, Array
from jax.nn import leaky_relu

class Layer(NamedTuple):
    w: Array
    b: Array
    
    def __call__(self, x):
        return self.w @ x + self.b

class Model(NamedTuple):
    in_layer: Layer
    h_layer: Layer
    out_layer: Layer

    def __call__(self, x):
        y1 = leaky_relu(self.in_layer(x))
        y2 = leaky_relu(self.h_layer(y1))
        y3 = self.out_layer(y2)
        return y3

def init_layer(key, in_feat, out_feat):
    wkey, bkey = random.split(key)
    return Layer(
        w=random.normal(wkey, (out_feat, in_feat)) * .01,
        b=random.normal(bkey, (out_feat,)) * .01)

def init_model(key, in_feat, h_feat, out_feat):
    ikey, hkey, okey = random.split(key, 3) # haha "okey"
    return Model(
        in_layer=init_layer(ikey, in_feat, h_feat),
        h_layer=init_layer(hkey, h_feat, h_feat),
        out_layer=init_layer(okey, h_feat, out_feat))
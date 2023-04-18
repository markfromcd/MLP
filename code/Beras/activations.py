import numpy as np

from .core import Diffable


class LeakyReLU(Diffable):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        # TODO: Given an input array `x`, compute LeakyReLU(x)
        self.inputs = inputs
        # Your code here:
        max = np.maximum(self.inputs,0)
        min = np.minimum(self.inputs,0)*self.alpha
        
        self.outputs = max+min
        return self.outputs

    def input_gradients(self):
        # TODO: Compute and return the gradients
        dpos = np.where(self.inputs>0,1,0)
        dneg = np.where(self.inputs<0, self.alpha,0)
        grad = dpos + dneg
        return grad

    def compose_to_input(self, J):
        # TODO: Maybe you'll want to override the default?
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied
        """
        input_gradient = self.input_gradients()
        compose_to_input = input_gradient * J
        return compose_to_input
        #return super().compose_to_input(J)


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)


class Softmax(Diffable):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """Softmax forward pass!"""
        # TODO: Implement
        # HINT: Use stable softmax, which subtracts maximum from
        # all entries to prevent overflow/underflow issues
        self.inputs = inputs
        # Your code here:
        exp = np.exp(self.inputs - np.max(self.inputs, axis=-1, keepdims=True))
        self.outputs = exp / (np.sum(exp, axis=-1, keepdims=True) )
        #self.outputs = None
        return self.outputs

    def input_gradients(self):
        """Softmax backprop!"""
        # TODO: Compute and return the gradients
        num_features = self.inputs.shape[1]

        gradient = np.zeros((self.inputs.shape[0], self.inputs.shape[-1], self.inputs.shape[-1]))
        tmp = np.expand_dims(self.outputs[0],1)
        num = len(self.inputs)
        for i in range(num):
            np.fill_diagonal(gradient[i], 1)
            tmp = np.expand_dims(self.outputs[i],1)
            tmp = np.tile(tmp,(1,num_features))
            
            gradient[i] -= tmp
            gradient[i] *= tmp.T
        return gradient

import numpy as np
import tensorflow as tf

from .core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        """Categorical accuracy forward pass!"""
        super().__init__()
        # TODO: Compute and return the categorical accuracy of your model given the output probabilities and true labels
        #metric = tf.keras.metrics.CategoricalAccuracy()
        #metric.update_state(labels,probs)
        #categorical_accuracy = metric.result().numpy()
        
        #return 
        return np.mean(np.argmax(probs, axis=1) == np.argmax(labels, axis=1))
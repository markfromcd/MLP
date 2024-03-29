from types import SimpleNamespace

import Beras
import numpy as np


class SequentialModel(Beras.Model):
    """
    Implemented in Beras/model.py

    def __init__(self, layers):
    def compile(self, optimizer, loss_fn, acc_fn):
    def fit(self, x, y, epochs, batch_size):
    def evaluate(self, x, y, batch_size):           ## <- TODO
    """

    def call(self, inputs):
        """
        Forward pass in sequential model. It's helpful to note that layers are initialized in Beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        # TODO: The call function!
        x = np.copy(inputs)
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_step(self, x, y, training=True):
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! 
        Most of this method (forward, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        # TODO: Compute loss and accuracy for a batch.
        # If training, then also update the gradients according to the optimizer
        #preds = None
        #loss = None
        with Beras.GradientTape() as tape:
            preds = self.call(x)
            loss = self.compiled_loss(preds,y)
            
        grads = tape.gradient()
        if training:
            self.optimizer.apply_gradients(self.trainable_variables, grads)
            
        accuracy = self.compiled_acc(preds,y)
        return {"loss": loss, "acc": accuracy}


def get_simple_model_components():
    """
    Returns a simple single-layer model.
    """
    ## DO NOT CHANGE IN FINAL SUBMISSION

    from Beras.activations import Softmax
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer

    # TODO: create a model and compile it with layers and functions of your choice
    model = SequentialModel([Dense(784, 10), Softmax()])
    model.compile(
        optimizer=BasicOptimizer(0.02),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=100)


def get_advanced_model_components():
    """
    Returns a multi-layered model with more involved components.
    """
    from Beras.activations import Softmax, LeakyReLU
    from Beras.layers import Dense
    from Beras.losses import CategoricalCrossentropy
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import Adam
    # TODO: create/compile a model with layers and functions of your choice.
    # model = SequentialModel()
    model = SequentialModel([Dense(784, 256), LeakyReLU(), Dense(256, 10),Softmax()])
    model.compile(
        optimizer=Adam(0.004),
        loss_fn=CategoricalCrossentropy(),
        acc_fn=CategoricalAccuracy(),
    )

    return SimpleNamespace(model=model, epochs=10, batch_size=100)


if __name__ == "__main__":
    """
    Read in MNIST data and initialize/train/test your model.
    """
    from Beras.onehot import OneHotEncoder
    import preprocess

    ## Read in MNIST data,
    train_inputs, train_labels = preprocess.get_data_MNIST("train", "../data")
    test_inputs,  test_labels  = preprocess.get_data_MNIST("test",  "../data")

    ## TODO: Use the OneHotEncoder class to one hot encode the labels
    #ohe = lambda x: 0  ## placeholder function: returns zero for a given input
    ohe = OneHotEncoder()

    ## Get your model to train and test
    simple = False
    args = get_simple_model_components() if simple else get_advanced_model_components()
    model = args.model

    ## REMINDER: Threshold of accuracy: 
    ##  1470: >85% on testing accuracy from get_simple_model_components
    ##  2470: >95% on testing accuracy from get_advanced_model_components

    # Fits your model to the training input and the one hot encoded labels
    # This does NOT need to be changed
    train_agg_metrics = model.fit(
        train_inputs, 
        ohe(train_labels), 
        epochs     = args.epochs, 
        batch_size = args.batch_size
    )

    ## Feel free to use the visualize_metrics function to view your accuracy and loss.
    ## The final accuracy returned during evaluation must be > 80%.

    # from visualize import visualize_images, visualize_metrics
    # visualize_metrics(train_agg_metrics["loss"], train_agg_metrics["acc"])
    # visualize_images(model, train_inputs, ohe(train_labels))

    ## Evaluates your model using your testing inputs and one hot encoded labels.
    ## This does NOT need to be changed
    test_agg_metrics = model.evaluate(test_inputs, ohe(test_labels), batch_size=100)
    print('Testing Performance:', test_agg_metrics)

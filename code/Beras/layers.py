import numpy as np

from .core import Diffable


class Dense(Diffable):
    def __init__(self, input_size, output_size, initializer="kaiming"):
        super().__init__()
        self.w, self.b = self.__class__._initialize_weight(
            initializer, input_size, output_size
        )
        self.weights = [self.w, self.b]
        self.inputs  = None
        self.outputs = None

    def forward(self, inputs):
        """Forward pass for a dense layer! Refer to lecture slides for how this is computed."""
        self.inputs = inputs

        # TODO: implement the forward pass and return the outputs
        #for w,b in self.wights:
        self.outputs = np.dot(self.inputs,self.w)+self.b
        return self.outputs

    def weight_gradients(self):
        """Calculating the gradients wrt weights and biases!"""
        # TODO: Implement calculation of gradients
        wgrads = np.expand_dims(self.inputs,axis=2)#np.dot(self.inputs.T, self.input_gradients())
        bgrads = np.ones((self.b.shape[0],))#np.sum(self.input_gradients(), axis=0)
        return wgrads, bgrads

    def input_gradients(self):
        """Calculating the gradients wrt inputs!"""
        # TODO: Implement calculation of gradients
        igrads = self.weights[0]#np.dot(self.input_gradients(), self.w.T)
        return igrads

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size):
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"
        io_size = (input_size, output_size)

        # TODO: Implement default assumption: zero-init for weights and bias
        w = None
        b = None
        if initializer == "zero":
            w = np.zeros(io_size)
            b = np.zeros(output_size)
        if initializer == "normal":
            w = np.random.normal(0, 1, io_size)
            #b = np.random.normal(0, 1, output_size)
            b = np.zeros(output_size)
        if initializer == "xavier":
            w = np.random.normal(0, np.sqrt(2 / (input_size + output_size)), io_size)
            b = np.zeros(output_size)
            #b = np.random.normal(0, np.sqrt(2 / (input_size + output_size)), output_size)
        if initializer == "kaiming":
            w = np.random.normal(0, np.sqrt(2 / input_size), io_size)
            #b = np.random.normal(0, np.sqrt(2 / input_size), output_size)
            b = np.zeros(output_size)
        # TODO: Implement remaining options (normal, xavier, kaiming initializations). Note that
        # strings must be exactly as written in the assert above

        return w, b

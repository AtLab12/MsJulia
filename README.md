# MsJulia

MsJulia contains code written as an attempt to implement a convolutional neural network based on a custom automatic differentiation solution. The solution works; however, performance-wise, it would greatly benefit from some work. The implemented network has the following architecture:

Architecture of the CNN consists of: <br />
• Convolutional layer<br />
• Max pooling layer<br />
• Flattened layer<br />
• Two dense layers<br />
• Loss function<br />

The convolutional layer employs a 3x3 kernel to process
inputs of dimensions 28x28x1, converting them from a single
channel to six output channels. This layer uses the ReLU
activation function and does not include a bias term.
Subsequently, a max pooling layer with a 2x2 window is
utilized to down sample the feature maps, thereby reducing
the computational load.
The flattening operation then converts the multi-dimensional
feature maps into a one-dimensional vector, facilitating the
transition from convolutional layers to dense layers.
The first dense layer comprises 84 neurons, without biases,
and also uses the ReLU activation function. The final layer is a dense layer consisting of 10 output units,
corresponding to the number of target classes. The activation
function for this layer is softmax, which transforms the raw
outputs of the neural network into a vector of probabilities.

![NNimage](https://github.com/AtLab12/MsJulia/assets/40431386/fe1e282f-3fd1-47d6-940d-d8895985b9cc)

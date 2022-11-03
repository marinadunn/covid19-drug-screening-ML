import  tensorflow  as      tf;

from    Network     import  Network;



def SSE_Loss(   U           : Network, 
                Inputs      : tf.Tensor, 
                Targets     : tf.Tensor, 
                Training    : bool      = False) -> tf.Tensor:
    """
    This function evaluates the sum of square 2-norm between U's predictions and the Targets. Here,
    the ith prediction is the image under U of the ith row of Inputs. Once we have the ith
    prediction, we find the 2-norm of it minus the ith row of Targets. We do this for each i. 
    We then square these values and sum those values. 

    -----------------------------------------------------------------------------------------------
    Arguments: 

    U: A neural network that maps from R^n to R^m. This function evaluates how well U maps rows of 
    Inputs (which are in R^n) to rows of Targets (which are in R^m).

    Inputs: A 2D tensor. Specifically, this should be an N by n tensor, where N is the number of 
    examples and n is the dimension of the domain of U.

    Targets: A 2D tensor. Specifically, this should be an N by m tensor, where N is the number of 
    examples and m is the dimension of the co-domain of U.

    Training: If true, calls U with training = True. This modifies how batch normalization and 
    dropout layers work.

    -----------------------------------------------------------------------------------------------
    Returns:
    
    A single element tensor whose lone entry holds the following value:
        \sum_{i = 1}^{N} ||U(Inputs[i, :]) - Targets[i, :]||_2^2
    """

    # First, pass the Inputs through U.
    Predictions : tf.Tensor = U(Inputs, training = Training);

    # Next, evaluate the difference between Predictions and Targets.
    Error : tf.Tensor = tf.subtract(Predictions, Targets);

    # Notice that \sum_{i = 1}^{N} ||U(Inputs[i, :]) - Targets[i, :]||_2^2 is equal to 
    #       \sum_{i = 1}^{N} \sum_{j = 1}^{m} |U(Inputs[i, :])[j] - Targets[i, j]|^2
    Loss : tf.Tensor = tf.reduce_sum(tf.multiply(Error, Error));

    # All done!
    return Loss;



def SAE_Loss(   U           : Network, 
                Inputs      : tf.Tensor, 
                Targets     : tf.Tensor, 
                Training    : bool      = False) -> tf.Tensor:
    """
    This function evaluates the sum of absolute error between U's predictions and the Targets. 
    Here, the ith prediction is the image under U of the ith row of Inputs. Once we have the ith
    prediction, we find the 1-norm of it minus the ith row of Targets. We do this for each i. We
    sum those values. 

    -----------------------------------------------------------------------------------------------
    Arguments: 

    U: A neural network that maps from R^n to R^m. This function evaluates how well U maps rows of 
    Inputs (which are in R^n) to rows of Targets (which are in R^m).

    Inputs: A 2D tensor. Specifically, this should be an N by n tensor, where N is the number of 
    examples and n is the dimension of the domain of U.

    Targets: A 2D tensor. Specifically, this should be an N by m tensor, where N is the number of 
    examples and m is the dimension of the co-domain of U.

    Training: If true, calls U with training = True. This modifies how batch normalization and 
    dropout layers work.

    -----------------------------------------------------------------------------------------------
    Returns:
    
    A single element tensor whose lone entry holds the following value:
        \sum_{i = 1}^{N} ||U(Inputs[i, :]) - Targets[i, :]||_1
    """

    # First, pass the Inputs through U.
    Predictions : tf.Tensor = U(Inputs, training = Training);

    # Next, evaluate the difference between Predictions and Targets.
    Error : tf.Tensor = tf.subtract(Predictions, Targets);

    # Notice that \sum_{i = 1}^{N} ||U(Inputs[i, :]) - Targets[i, :]||_2^2 is equal to 
    #       \sum_{i = 1}^{N} \sum_{j = 1}^{m} |U(Inputs[i, :])[j] - Targets[i, j]|_1
    Loss : tf.Tensor = tf.reduce_sum(tf.abs(Error));

    # All done!
    return Loss;



def L1_Loss(U : Network) -> tf.Tensor:
    """
    This function computes the L1 norm of U's parameters.

    -----------------------------------------------------------------------------------------------
    Arguments:

    U: A network object. We compute the L1 norm of its paramaters. To do this, we compute the 
    absolute value of each of U's weights and biases and then add them together.

    -----------------------------------------------------------------------------------------------
    Returns: 

    A single element tensoe whose lone element holds the L1 norm of U's parameters.
    """

    # Initialize the loss. 
    Loss : tf.Tensor = tf.zeros(shape = 1, dtype = tf.float32);

    # Add on the sum of the squares of the components of the weights/biases from U's layers.
    for i in range(len(U.Layers)):
        # First, get the weight matrix, bias vector for this layer.
        W : tf.Tensor = U.Layers[i].W;
        b : tf.Tensor = U.Layers[i].b;

        # Add the element-wise absolute value of their components to Loss.
        Loss += tf.reduce_sum(tf.abs(W));
        Loss += tf.reduce_sum(tf.abs(b));

    # All done!
    return Loss;



def L2_Loss(U : Network) -> tf.Tensor:
    """
    This function computes the square of the L2 norm of U's parameters.

    -----------------------------------------------------------------------------------------------
    Arguments:

    U: A network object. We compute the L2 norm of its paramaters. To do this, we compute the 
    square each of U's weights/biases, add them together, and then take the square root of the 
    result. 

    -----------------------------------------------------------------------------------------------
    Returns: 

    A single element tensoe whose lone element holds the L2 norm of U's parameters.
    """

    # Initialize the loss. 
    Loss : tf.Tensor = tf.zeros(shape = 1, dtype = tf.float32);

    # Add on the sum of the squares of the components of the weights/biases from U's layers.
    for i in range(len(U.Layers)):
        # First, get the weight matrix, bias vector for this layer.
        W : tf.Tensor = U.Layers[i].W;
        b : tf.Tensor = U.Layers[i].b;

        # Add the element-wise square of their components to Loss.
        Loss += tf.reduce_sum(tf.multiply(W, W));
        Loss += tf.reduce_sum(tf.multiply(b, b));

    # The loss now holds the square of the L2 norm of U's parameters. To get the L2 norm, we simply
    # need to take the square root.
    return Loss;


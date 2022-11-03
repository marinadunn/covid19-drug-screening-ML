import  tensorflow          as      tf;
import  numpy;
from    typing              import  List, Dict, Tuple;



class Affine(tf.keras.layers.Layer):
    """
    This class implements an Affine map from R^n to R^m. In particular, it maps the vector 
    x \in R^n to 
            x*W + b
    Here, W \in R^{n x m} is a matrix, and b \in R^m is a vector. Here, x*W is the vector whose ith
    component is the dot product of x and the ith column of W. We write it this way so that the
    user can pass a matrix of data. In particular, the user typically passes an n by K data matrix, 
    X, whose ith row holds the ith input. In this case, X*W + B holds the image under the affine 
    map of the ith input, where B is an K by m matrix whose ith row is b. 

    You may be wondering... why not just use Keras' dense class instead of defining your own? 
    Becuase I like writing my own code. That's why... 
    """

    def __init__(   self, 
                    Dim_In              : int,
                    Dim_Out             : int,
                    Weight_Initializer  : tf.keras.initializers.Initializer = tf.keras.initializers.GlorotUniform(),
                    Bias_Initializer    : tf.keras.initializers.Initializer = tf.keras.initializers.Zeros()):
            """ 
            ---------------------------------------------------------------------------------------
            Arguments:

            Dim_In, Dim_Out: Recall that this class defines an affine map from R^n to R^m. Dim_In 
            is n and Dim_Out is m.

            Weight_Initializer: A keras initializer object to intiailize the weight matrix. Xavier 
            uniform by default. 

            Bias_Initializer: A keras initializer object to initialize the bias vector. Zero by 
            default.
            """

            # First, call the Layer class initializer. 
            super(Affine, self).__init__();

            # Check for valid inputs
            assert(Dim_In   > 0);
            assert(Dim_Out  > 0);

            # Next, set the dimensions.
            self.Dim_In     = Dim_In;
            self.Dim_Out    = Dim_Out;

            # Now... set up the weight matrix. 
            self.W = tf.Variable(initial_value = Weight_Initializer(shape = (Dim_In, Dim_Out), 
                                                                    dtype = tf.float32));
            
            # Finally, set up the bias vector.
            self.b = tf.Variable(initial_value = Bias_Initializer(  shape = Dim_Out, 
                                                                    dtype = tf.float32));
                                
    
    def call(   self, 
                X       : tf.Tensor) -> tf.Tensor:
        """ 
        This function maps X \in R^{K x n} to a matrix Y \in R^{K x m} whose i,j entry is given by
                X[i, :] \cdot W[:, j] + b_j
        In other words, the ith row of Y holds the image under the affine map of the ith row of X.

        -------------------------------------------------------------------------------------------
        Arguments: 

        X: This should be an N by n (2D) tensor, where n = Dim_In. The ith row of this matrix holds 
        the ith input to the affine map. 

        -------------------------------------------------------------------------------------------
        Returns:   

        Y: A tensor whose ith row holds the image under the affine map, x -> x*W + b, of the ith 
        row of X. 
        """

        # First, compute XW
        WX : tf.Tensor = tf.matmul(X, self.W);

        # Add the bias to each row and return!
        return tf.nn.bias_add(WX, self.b);



class Network(tf.keras.Model):
    """         
    This class implements a neural network with a user-specified width at each layer. We also 
    allow the user to control the activation function on the hidden and output layers. 
    
    Network objects are defined by the Widths argument, which specifies the number of neurons in 
    each of the network's layers. This includes the input and output layers!! 
    """

    def __init__(   self, 
                    Widths              : List[int], 
                    Hidden_Activation   : str   = "elu", 
                    Output_Activation   : str   = "none"):
        """ 
        -------------------------------------------------------------------------------------------
        Arguments: 

        Widths: A list of integers. If this list has N + 2 entries, then the network will have 1 
        input layer, N Hidden layers, and 1 output layer. The ith entry of Widths  specifies the 
        width (number of neurons) of the ith layer (with the 0th layer being the input). Thus, 
            - entry 0 specifies the input dimension (dimension of the networks' domain).
            - entries 1, ... , N specify the widths of the hidden layers
            - entry N + 1 specifies the output dimension (dimension of the network's co-domain). 

        Hidden_Activation: A string specifying which activation function the hidden layers should
        use. Currently, we allow sigmoid, tanh, relu, and elu. Default is elu.

        Output_Activation: A strining specifying which activation function to use on the output 
        layer. By default, this is set to None. Currently, we allow none, sigmoid, tanh, relu, elu,
        and softmax. 
        """

        # First thing is first... call the Model initializer. 
        super(Network, self).__init__();

        # Now, determine the network depth.
        self.Num_Hidden_Layers : int = len(Widths) - 2; 

        # Set network input/output dimension and Widths.
        self.Dim_In     = Widths[0];
        self.Dim_Out    = Widths[-1];
        self.Widths     = Widths;

        # Now, let's determine the hidden activation function. We could pass the argument directly
        # to the Dense initializer, but I want to restrict which activation functions the user can
        # use. 
        if  (Hidden_Activation.lower().strip() == "sigmoid"):
            self.Hidden_Activation = tf.keras.activations.sigmoid;
        elif(Hidden_Activation.lower().strip() == "tanh"):
            self.Hidden_Activation = tf.keras.activations.tanh;
        elif(Hidden_Activation.lower().strip() == "elu"):
            self.Hidden_Activation = tf.keras.activations.relu;
        elif(Hidden_Activation.lower().strip() == "relu"):
            self.Hidden_Activation = tf.keras.activations.elu;
        else:
            print("Error: Unrecognized hidden activation function. Raised by Network initializer.");
            print("Got %d but expected \"sigmoid\", \"tanh\", \"relu\", or \"elu\". Using elu." % Hidden_Activation);
            self.Hidden_Activation = tf.keras.activations.elu;

        # Next, let's determine the output activation (if there is one).
        if  (Output_Activation.lower().strip() == "none"):
            self.Output_Activation = None;
        elif(Output_Activation.lower().strip() == "sigmoid"):
            self.Output_Activation = tf.keras.activations.sigmoid;
        elif(Output_Activation.lower().strip() == "tanh"):
            self.Output_Activation = tf.keras.activations.tanh;
        elif(Output_Activation.lower().strip() == "elu"):
            self.Output_Activation = tf.keras.activations.elu;
        elif(Hidden_Activation.lower().strip() == "elu"):
            self.Hidden_Activation = tf.keras.activations.relu;
        elif(Output_Activation.lower().strip() == "softmax"):
            self.Output_Activation = tf.keras.activations.softmax;
        else:
            print("Error: Unrecognized hidden activation function. Raised by Network initializer.");
            print("Got %d but expected \"sigmoid\", \"tanh\", \"relu\", \"elu\", or \"softmax\". Using none." % Output_Activation);
            self.Output_Activation = None;

        # Finally, set the affine parts of each layer. 
        self.Layers = [];

        for i in range(self.Num_Hidden_Layers + 1):
            Layer_i  = Affine(  Dim_In  = Widths[i],
                                Dim_Out = Widths[i + 1]);
            
            self.Layers.append(Layer_i);


    def call(   self, 
                X       : tf.Tensor) -> tf.Tensor:
            """
            A network object represents a map from R^n to R^m. This function defines how that map
            works.

            ---------------------------------------------------------------------------------------
            Arguments: 

            X : A 2D tensor. Each row of X represents an input to the network. 

            ---------------------------------------------------------------------------------------
            Returns: 

            A 2D tensor whose ith entry holds the image under the network of the ith row of X.
            """

            # Pass X through the Hidden layers. On the ith step, we apply the ith affine map, and 
            # then apply the activation function.
            for i in range(self.Num_Hidden_Layers):
                X = self.Hidden_Activation(self.Layers[i](X));

            # Apply the last affine map.
            X = self.Layers[-1](X);

            # Apply the output activation, if there is one.
            if(self.Output_Activation is not None):
                X = self.Output_Activation(X);

            # All done!
            return X;
        
    

    def Copy(self):
        """
        This function returns a deep copy of self.
        
        -------------------------------------------------------------------------------------------
        Returns:

        A Network object that is, for all intents and purposes, an identical copy of self, but 
        stored in a different location. Specifically, the returned network has the same 
        architecture, weights, and biases of self, but its weights/biases are stored in a different
        location in memory. As such, changes to either networks weights/biases will not impact the 
        others.
        """

        # First, initialize a new network object. 
        Copy = Network(Widths   = self.Widths);

        # Copy over activation functions.
        Copy.Hidden_Activation  = self.Hidden_Activation;
        Copy.Output_Activation  = self.Output_Activation;

        # Now copy over the weights and biases
        for i in range(len(self.Layers)):
            Copy.Layers[i].W.assign(self.Layers[i].W);
            Copy.Layers[i].b.assign(self.Layers[i].b);

        # All done!
        return Copy;
            



class Bagging_Network(Network):
    """
    A Bagging Network trains on a subset of the features in some data. Each object is characterized
    by a "Feature Subset" and an architecture. A bagging network takes in data in R^N, for some N.
    It then selects a subsets of the components of the inputs and feeds those through the 
    underlying network. The Feature_Subset attribute specifies which components of the inputs it 
    uses. 

    The Withs, Hidden_Activation, and Output_Activation initializer arguments define the 
    architecture of the underlying Network, just like a regular Network object. 
    """
    def __init__(   self, 
                    Dim_In              : int,
                    Feature_Subset, 
                    Widths              : List[int],
                    Hidden_Activation   : str       = "elu",
                    Output_Activation   : str       = "none"):
        """
        -------------------------------------------------------------------------------------------
        Arguments:

        Dim_In: A Bagging network trains on a subset of the features in some data. Dim_In specifies 
        the number of features in that data. 

        Feature_Subset: This is a List-like object whose ith entry specifies the ith feature of the 
        data that the Bagging_Network should accept as input. The length of this list should match 
        Widths[0]. Further, each element of Feature_Subset should be in the set
        {0, 1, ... , Dim_In - 1}. 

        Widths: A list of integers. The ith entry specifies the number of neurons in the ith 
        layer of the network. This includes the domain and co-domain. Thus, if this list has N + 2 
        entries, the network will have N hidden layers (Widths[0] is input dimension, Widths[1] is 
        output dimension). Widths[0] must match len(Feature_Subset).

        Hidden_Activation: A string specifying which activation function the hidden layers should
        use. 

        Output_Activation: A strining specifying which activation function to use on the output 
        layer. 
        """


        # First, make sure the inputs are valid.
        assert(Dim_In > 0);

        for i in range(len(Feature_Subset)):
            assert(Feature_Subset[i] >= 0);
            assert(Feature_Subset[i] < Dim_In);

        for i in range(len(Widths)):
            assert(Widths[i] > 0);

        assert(Widths[0] == len(Feature_Subset));

        # Call the Network initializer.
        super(Bagging_Network, self).__init__(
                                Widths              = Widths,
                                Hidden_Activation   = Hidden_Activation,
                                Output_Activation   = Output_Activation);

        # Set up attributes. Note that we override the Dim_In set by the Network initializer.
        self.Dim_In         : int       = Dim_In;
        self.Feature_Subset : tf.Tensor = tf.reshape(tf.constant(Feature_Subset, dtype = tf.int32), (-1));    

    

    def call(   self, 
                Inputs  : tf.Tensor):
        """
        This passes the columns of Inputs specified by the Subset attribute through the 
        network.

        ---------------------------------------------------------------------------------------
        Arguments: 

        X : A 2D tensor. Each row of X represents an input to the network. 

        ---------------------------------------------------------------------------------------
        Returns: 

        A 2D tensor whose ith entry holds the image under the network of the ith row of X.        
        """

        # Select the features that this model uses.
        X : tf.Tensor = tf.gather( 
                            params  = Inputs,
                            indices = self.Feature_Subset, 
                            axis    = 1);
        
        # Evaluate the model.
        return super(Bagging_Network, self).call(X);



    def Copy(self):
        """
        This function returns a deep copy of self.
        
        -------------------------------------------------------------------------------------------
        Returns:

        A Bagging_Network object that is, for all intents and purposes, an identical copy of self, 
        but stored in a different location. Specifically, the returned network has the same 
        architecture, weights, and biases of self, but its weights/biases are stored in a different
        location in memory. As such, changes to either networks weights/biases will not impact the 
        others.        
        """

        # First, a new Bagging Network object.
        Copy = Bagging_Network( Dim_In          = self.Dim_In,
                                Feature_Subset  = self.Feature_Subset,
                                Widths          = self.Widths);

        # Copy over activation functions.
        Copy.Hidden_Activation  = self.Hidden_Activation;
        Copy.Output_Activation  = self.Output_Activation;

        # Now copy over the weights and biases
        for i in range(len(self.Layers)):
            Copy.Layers[i].W.assign(tf.identity(self.Layers[i].W));
            Copy.Layers[i].b.assign(tf.identity(self.Layers[i].b));

        # All done!
        return Copy;
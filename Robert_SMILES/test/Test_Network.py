# Nonsense to add Code diectory to the python search path.
import os;
import sys;

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the src directory to the python path.
src_Path   = os.path.join(parent_dir, "src");
sys.path.append(src_Path);

import  tensorflow  as      tf;
import  unittest;
import  random;
import  numpy;
from    typing      import  List;

from    Network     import Affine, Network;



Verbose : bool = True;



class Test_Network(unittest.TestCase):
    def test_Affine(self):
        ###########################################################################################
        # Setup.

        # Pick an input and output dimension.
        Dim_In  : int = random.randint(10, 200);
        Dim_Out : int = random.randint(10, 200);
        
        # First, generate a random input matrix. 
        N_Inputs : int = random.randint(100, 1000);

        X : tf.Tensor = tf.keras.initializers.RandomUniform(
                            minval = -1., 
                            maxval = 1.)(   shape = (N_Inputs, Dim_In), 
                                            dtype = tf.float32);

        if(Verbose):
            print("Dim_In   = %u" % Dim_In);
            print("Dim_Out  = %u" % Dim_Out);
            print("N_Inputs = %u" % N_Inputs);


        ###########################################################################################
        # Test 0: Zero map.

        # Now, set up a simple "zero" affine map (use zero initializer).
        f0 = Affine(Dim_In  = Dim_In,
                    Dim_Out = Dim_Out, 
                    Weight_Initializer = tf.keras.initializers.Zeros());

        # Now, check that f0 is indeed a zero map. To check this, we sum the absolute values of the
        # entries of the image of X under f0. This sum is zero if and only if X = 0.
        Y           : tf.Tensor = f0(X);
        Sum_Abs_Y   : float     = tf.reduce_sum(tf.abs(Y));

        self.assertEqual(Sum_Abs_Y, 0.);

        # Also make sure the dimension of Y is N_Inputs x Dim_Out.
        self.assertEqual(Y.shape[0], N_Inputs);
        self.assertEqual(Y.shape[1], Dim_Out);


        ###########################################################################################
        # Test 1: Ones map.
        
        # Now, set up a "ones" map (each component of weight matrix, bias vector is 1).
        f1 = Affine(Dim_In  = Dim_In,
                    Dim_Out = Dim_Out, 
                    Weight_Initializer  = tf.keras.initializers.Ones(),
                    Bias_Initializer    = tf.keras.initializers.Ones());
            
        # Pass X through f1.
        Y           : tf.Tensor = f1(X);

        # set up some tolerance to account for floating point roundoff.
        eps : float = 1e-5;

        # Check that f is indeed a "ones map". If this is the case, then the i,j entry of Y should 
        # equal the sum of the components of the ith row of X, plus 1. 
        for i in range(N_Inputs):
            # The target value is the same for each entry in the ith row.
            Target : float = tf.reduce_sum(X[i, :]) + 1.;

            # Check that each entry in the ith row of Y is within epsilon of the Target.
            self.assertEqual(numpy.sum(tf.greater(tf.abs(Y[i, :] - Target), eps).numpy()).item(), 0.);



    def test_Network(self):
        ###########################################################################################
        # Setup 

        # First, determine the number of layers.
        Num_Layers : int = random.randint(2, 20);

        # Now determine the width of each layer.
        Widths : List[int] = [];
        for i in range(Num_Layers):
            Widths.append(random.randint(10, 100));
        
        if(Verbose):
            print(Widths);
        
        # Set up a network.
        U : Network = Network(Widths = Widths);


        ###########################################################################################
        # Test 0: Number of layers.

        self.assertEqual(len(U.Layers), Num_Layers - 1);


        ###########################################################################################
        # Test 1: Affine maps

        # Second, let's make sure each affine map has the correct domain and co-domain. In 
        # particular, there should be Num_Layers - 1 affine maps, and the ith one should map 
        # from R^n(i) to R^n(i + 1), where n(i) = Widths[i].
        for i in range(Num_Layers - 1):
            # Fetch ith affine map.
            Layer : Affine = U.Layers[i];

            self.assertEqual(Layer.Dim_In,  Widths[i]);
            self.assertEqual(Layer.Dim_Out, Widths[i + 1]);

        
        
if(__name__ == "__main__"):
    unittest.main();
import  tensorflow  as      tf;
from    typing      import  List;

from    Network     import  Network;



class Logistic(Network):
    """ 
    This class defines a logistic regression classifier. It is a subclass of the Network class. 
    This class allows for both binary and multi-class logistic regression classifiers. By default, 
    objects of this class are binary classifiers.

    A logistic regression classifier has 0 hidden layers (only an input and output layer). The 
    output layer uses the sigmoid (if binary) or softmax (if multi-class) activation function.

    Objects of this class are defined by their input dimension and the number of target classes.
    If the number of target classes is 2 (it must be at least 2!), then this is a binary classifier
    and the co-domain is R. If the number of target classes is n > 2, then this is a multi-class 
    classifier and the co-domain is R^n.
    """

    def __init__(   self, 
                    Dim_In              : int, 
                    Num_Target_Classes  : int   = 2):
            """ 
            ---------------------------------------------------------------------------------------
            Arguments: 

            Dim_In: Every machine learning model defines a map from R^n to R^m. This is n.

            Num_Target_Classes: A logistic regression map acts as a classifier. As such, objects 
            of this class should... well... classify things. This is the number of classes that 
            the inputs can belong to. If this number is 2, we set up a binary classifier. If it is
            more than 2, then we set up a multi-class classifier.

            Normalize_Input: If true, we add a batch normalization layer before the affine map.
            """

            # Num_Target_Classes must be at least 2.
            assert(Num_Target_Classes >= 2);

            # First, determine if we are a binary or multi-class classifier. 
            if(Num_Target_Classes == 2): # Binary
                # The co-domain is 1 in this case.
                Dim_Out : int = 1;

                # Set up weights list.
                Widths  : List[int] = [Dim_In, Dim_Out];

                # Call super class initializer.
                super(Logistic, self).__init__( Widths              = Widths,
                                                Output_Activation   = "sigmoid");
            else: # Multi-class
                # The co-domain is R^n, where n = Num_Target_Classes.
                Dim_Out : int = Num_Target_Classes;
            
                # Set up the width s list.
                Widths  : List[int] = [Dim_In, Dim_Out];

                # Call the super class initializer.
                super(Logistic, self).__init__( Widths              = Widths, 
                                                Output_Activation   = "softmax");
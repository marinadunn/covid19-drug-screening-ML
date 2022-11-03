from multiprocessing.sharedctypes import Value
import  tensorflow      as      tf;
import  numpy;
from    typing          import  List;

from    Subset          import  Generate_Index_Subsets;
from    Test_Train      import  Train, Test_Binary_Classifier;
from    Network         import  Network;
from    Loss            import  SSE_Loss;



class Ensemble(tf.keras.Model):
    """
    This class defines a Ensemble of networks, all with the same domain and co-domain. To 
    initialize an Ensemble object, you must pass a list of pre-initialized Network objects (the 
    Ensemble's Sub-Model). This class basically acts as a wrapper for that list, making it easier 
    to train and evaluate the list as a single entity. 

    When evaluating an Ensemble object, you can specify how the call method works using the
    "Set_Call_Mode" method, or by specifying the "Call_Mode" initializer argument . In 
    particular, the user can choose to evaluate in "average" or "voting" mode. In average mode, we 
    eavauate each model on the input data and then report the average output. Voting mode, which 
    currently only works for binary classifiers, works as follows: We evaluate each sub-model on 
    the  input. Then, for each input, we count the number of sub-models whose output is greater 
    than .5. If less than half the models predict a value greater than 0.5, we say the classifiers 
    predict the input is in class 1. Otherwise, we say they predict the input is in class 2. 

    You can also specify which models should be used when calling the Ensemble. Every ensemble
    object has an attribute called "Call_Models". This is a list of integers. Each element of 
    this list is in the set {0, 1, ... , N - 1}, where N is the number of sub-models. If the number 
    i is in the Call_Models list, then model i will get a vote when computing the ensemble's 
    output (if in voting mode) or model i will be used to compute the average output (if in average 
    mode). Note that the Call_Models list can have repeated indicies (if you want to give a 
    certian model multiple votes or extra wegith in the average).
    """

    def __init__(   self,
                    Sub_Models      : List[Network],
                    Call_Mode       : str           = "average") -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:

        Sub_Models: A list of pre-compiled Network objects. These are the ensemble's sub-models. 

        Call_Mode: The mode we use in the Ensembel's "call" method. This can be "average" or 
        "vote". Note that "vote" mode only works if this is an ensemble of binary classifiers. 
        """

        super(Ensemble, self).__init__();

        # First, check that Sub_Models have the same domain and co-domain.
        Num_Sub_Models  : int = len(Sub_Models);
        Dim_In          : int = Sub_Models[0].Dim_In;
        Dim_Out         : int = Sub_Models[0].Dim_Out;

        for i in range(1, Num_Sub_Models):
            assert(Sub_Models[i].Dim_In     == Dim_In);
            assert(Sub_Models[i].Dim_Out    == Dim_Out);

        # Second, assign input dimension and sub-model attributes. 
        self.Dim_In         : int       = Dim_In;
        self.Dim_Out        : int       = Dim_Out;
        self.Num_Sub_Models : int       = Num_Sub_Models;
        self.Sub_Models     : List[Network] = Sub_Models;

        # Third, indicate that we want to use all sub-models when calling the ensemble.
        self.Call_Models        : List  = numpy.array([*range(Num_Sub_Models)]);
        self.Num_Call_Models    : int   = Num_Sub_Models;

        # Third, call the Model class initializer.
        super(Ensemble, self).__init__();

        # Forth, let's set the Call Mode. 
        if  (Call_Mode.lower()[0] == "a"):
            self.Call_Mode = "average";
        elif(Call_Mode.lower()[0] == "v"):
            self.Call_Mode = "vote";
        else:
            raise ValueError("Unknown call mode. Expected \"average\" or \"vote\"; got \"%s\"" % Call_Mode);



    def Set_Call_Mode(self, Call_Mode : str) -> None:
        """
        This function sets Ensemble's mode when using the call method.

        -------------------------------------------------------------------------------------------
        Arguments: 

        Call_Mode: A string specifying the call mode. This can either be "average" or "vote". Note
        that "vote" mode only works if this is an ensemble of binary classifiers. 
        """

        if  (Call_Mode.lower()[0] == "a"):
            self.Call_Mode = "average";
        elif(Call_Mode.lower()[0] == "v"):
            self.Call_Mode = "vote";
        else:
            raise ValueError("Unknown call mode. Expected \"average\" or \"vote\"; got \"%s\"" % Call_Mode);



    def Set_Call_Models(self, Call_Models) -> None:
        """ 
        This function sets which of the Ensembel's sub-models are used in the "call" method.

        -------------------------------------------------------------------------------------------
        Arguments:

        Call_Models: A list-like object of indices of the models you want to include when computing 
        the ensemble output. If the ensemble has N sub-models, then each element of Call_Models 
        should be between 0 and N - 1. 
        """

        # Call_Models must contain at least one model.
        assert(len(Call_Models) > 0);

        # Check that the entries of Call_Models are valid.
        for i in range(len(Call_Models)):
            assert(Call_Models[i] >= 0);
            assert(Call_Models[i] < len(self.Sub_Models));
        
        # Call_Models is valid... set it.
        self.Call_Models        = Call_Models;
        self.Num_Call_Models    = len(Call_Models);



    def call(   self,  
                Inputs  : tf.Tensor) -> tf.Tensor:
        """
        Evaluates the ensemble on the Inputs. We only use the models in the Call_Models list when 
        evaluating the output.

        -------------------------------------------------------------------------------------------
        Arguments:

        Inputs: This is a 2D tensor. The number of columns in this tensor should match the input 
        dimension.

        -------------------------------------------------------------------------------------------
        Returns: 

        A 2D tensor whose i,j entry holds the jth component of the ensemble's prediction for the
        ith input. If we are using "average" mode, then the i,j entry is the average of the i,j 
        outputs from each sub-model. If we are using "vote" mode, then this represents which class
        got the most "votes" (see above).
        """

        # Make sure the input is valid.
        assert(len(Inputs.shape)    == 2);

        if(self.Call_Mode == "average"):
            # Initialize "Sum_Predictions" using the predictions from  the first sub-model in the 
            # Call_Models list.
            Sum_Predictions : numpy.ndarray = self.Sub_Models[self.Call_Models[0]](Inputs);

            # Add on predictions from other sub-models in the Call_Models list. 
            for i in range(1, self.Num_Call_Models):
                Sum_Predictions += self.Sub_Models[self.Call_Models[i]](Inputs);

            # Average!
            return (1./float(self.Num_Call_Models))*Sum_Predictions;
        
        else:   # voting mode.
            # Initialize "total votes" using the predictions from the first sub-model in the 
            # Call_Models list. 
            Total_Votes : numpy.ndarray = tf.cast(tf.math.greater(self.Sub_Models[self.Call_Models[0]](Inputs), 0.5), dtype = tf.float32);

            # Add on the votes from the other sub-models in the Call_Models list.
            for i in range(1, self.Num_Call_Models):
                Total_Votes += tf.cast(tf.math.greater(self.Sub_Models[self.Call_Models[i]](Inputs), 0.5), dtype = tf.float32);

            # If more than half the models vote for a particular class, then the ensemble predicts 
            # that class
            Votes   : numpy.ndarray = tf.cast(tf.math.greater(Total_Votes, float(self.Num_Call_Models/2.)), dtype = tf.float32);  
            return Votes;
        


    def Rank_Sub_Models(    self, 
                            Inputs      : tf.Tensor,
                            Targets     : tf.Tensor,
                            Batch_Size  : int       = 64) -> tf.Tensor:
        """
        This function determines which of the ensemble's sub-models do the best job of mapping the
        Inputs to the corresponding targets. It returns a 1D tensor listing the ranks (in decending
        order) of the sub-models, according to their loss in mapping Inputs to Targets.

        -------------------------------------------------------------------------------------------
        Arguments:

        Inputs: A 2D tensor. Specifically, this should be an N by n tensor, where N is the number of 
        examples and n is the dimension of the domain of U.

        Targets: A 2D tensor. Specifically, this should be an N by m tensor, where N is the number of 
        examples and m is the dimension of the co-domain of U.

        -------------------------------------------------------------------------------------------
        Returns:

        A 1D numpy.ndarray (with int32 type) whose ith element holds the index of the sub-model 
        that produced the ith lowest loss.
        """

        # Initialize a tensor whose ith entry will hold the loss of the ith model.
        Sub_Model_Losses : numpy.ndarray = numpy.zeros(shape = self.Num_Sub_Models, dtype = numpy.float32);

        # Fetch number of examples.
        N : int = Inputs.shape[0];

        # Cycle through the sub-models.
        for m in range(self.Num_Sub_Models):
            # cycle through mini-batches (main loop)
            for i in range(0, N - Batch_Size, Batch_Size):
                # Get this batch of inputs, targets.
                Inputs_Batch    : tf.Tensor = Inputs[ i:(i + Batch_Size), :];
                Targets_Batch   : tf.Tensor = Targets[i:(i + Batch_Size), :];

                # Accumulate loss.
                Sub_Model_Losses[m] += SSE_Loss( 
                                        U       = self.Sub_Models[m],
                                        Inputs  = Inputs_Batch,
                                        Targets = Targets_Batch).numpy().item();

            # Final batch (clean up).
            Inputs_Batch    : tf.Tensor = Inputs[ (i + Batch_Size):, :];
            Targets_Batch   : tf.Tensor = Targets[(i + Batch_Size):, :];

            # Accumulate loss.
            Sub_Model_Losses[m] += SSE_Loss( 
                                    U       = self.Sub_Models[m],
                                    Inputs  = Inputs_Batch,
                                    Targets = Targets_Batch).numpy().item();

        return tf.argsort(Sub_Model_Losses, direction = "ASCENDING");
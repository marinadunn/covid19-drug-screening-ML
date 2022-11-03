import  tensorflow          as      tf;
import  numpy;
import  math;
from    typing              import  List, Tuple, Dict;


from    Test_Train          import  Train, Test, Test_Binary_Classifier;
from    Subset              import  Generate_Index_Subsets;
from    Network             import  Network, Bagging_Network;
from    Logistic            import  Logistic;
from    Ensemble            import  Ensemble;
from    Test_Train_Ensemble import  Train_Ensemble, Test_Binary_Classifier_Ensemble;



Verbose : bool = True;



def main_Ensemble():
    ###############################################################################################
    # Settings.

    # Ensemble Settings.
    Num_Models          : int           = 50;       # Number of models in the ensemble.
    Subset_Size         : int           = 80;       # Number of features each model trains on.

    # Sub-Model Settings 
    Widths              : List  = [Subset_Size, 40, 20, 10, 1];
    Hidden_Activation   : str   = "elu";
    Output_Activation   : str   = "sigmoid";

    # Optimizer settings.
    lr                  : float = 0.005;
    Optimizer                   = tf.keras.optimizers.Adam(learning_rate = lr);

    # Training settings.
    Num_Epochs          : int   = 100; 
    Normalizer          : str   = "L2";
    Lambda              : float = 0.001;


    ###############################################################################################
    # Setup Datasets

    # First, load the data sets. 
    DataSet         = numpy.load("../data/Cleaned_Data.npz");

    # Fetch the training/testing/validation inputs/targets.
    Train_Inputs    : tf.Tensor = tf.constant(DataSet["Train_Inputs"],  dtype = tf.float32);
    Train_Targets   : tf.Tensor = tf.constant(DataSet["Train_Targets"], dtype = tf.float32);

    Val_Inputs      : tf.Tensor = tf.constant(DataSet["Valid_Inputs"],  dtype = tf.float32);
    Val_Targets     : tf.Tensor = tf.constant(DataSet["Valid_Targets"], dtype = tf.float32); 

    Test_Inputs     : tf.Tensor = tf.constant(DataSet["Test_Inputs"],   dtype = tf.float32);
    Test_Targets    : tf.Tensor = tf.constant(DataSet["Test_Targets"],  dtype = tf.float32); 


    ###############################################################################################
    # Normalize the features. 

    # First, combine the training datasets.
    Data : tf.Tensor = tf.concat(   values      = [Train_Inputs, Val_Inputs, Test_Inputs],
                                    axis        = 0);

    # Compute each column's mean, std.
    Means   : tf.Tensor = tf.math.reduce_mean(Data, axis = 0, keepdims = True);
    Stds    : tf.Tensor = tf.math.reduce_std( Data, axis = 0, keepdims = True);

    # Now adjust each training set to have 0 mean and unit std.
    Train_Inputs    = tf.divide(tf.subtract(Train_Inputs, Means), Stds);
    Val_Inputs      = tf.divide(tf.subtract(Val_Inputs,   Means), Stds);
    Test_Inputs     = tf.divide(tf.subtract(Test_Inputs,  Means), Stds);
    

    ###############################################################################################
    # Initialize Model

    # Generate the feature subsets for the sub-models.
    Subsets = Generate_Index_Subsets(   Superset_Size   = Train_Inputs.shape[1],
                                        Subset_Size     = Subset_Size, 
                                        Num_Subsets     = Num_Models);
    
    # Set up the sub models. 
    Sub_Models = [];
    for i in range(Num_Models):
        Sub_Models.append(Bagging_Network(  
                                Dim_In              = Train_Inputs.shape[1], 
                                Feature_Subset      = Subsets[i, :],
                                Widths              = Widths, 
                                Hidden_Activation   = Hidden_Activation, 
                                Output_Activation   = Output_Activation));

    # Set up ensemble model. 
    Model = Ensemble(   Sub_Models  = Sub_Models,
                        Call_Mode   = "average");


    ###############################################################################################
    # Train the sub-models
    
    Train_Ensemble( Train_Inputs    = Train_Inputs, 
                    Train_Targets   = Train_Targets, 
                    Val_Inputs      = Val_Inputs, 
                    Val_Targets     = Val_Targets,
                    Ensemble_Model  = Model, 
                    Num_Epochs      = Num_Epochs,
                    Optimizer       = Optimizer,
                    Normalizer      = Normalizer,
                    Lambda          = Lambda);


    ###############################################################################################
    # Determine how many of the best-performing sub-models we should use.

    # Rank the models according to their accuracy on the validation set.
    Sub_Model_Ranks = Model.Rank_Sub_Models( 
                                Inputs  =   Val_Inputs,
                                Targets =   Val_Targets);

    if(Verbose):
        print("Sub-Model ranks: ", end = '');
        print((Sub_Model_Ranks + 1).numpy().tolist());
    
    # Evaluate the ensemble. Here, we perform a 1d parameter search over the number of sub-models.
    # We evaluate each possible number according its validation performance. 
    Max_Accuracy    : float = 0;
    Best_Num_Models : float = 0;
    for i in range(5, Num_Models + 1, 5):
        # Keep the i highest ranking models. 
        Model.Set_Call_Models(Sub_Model_Ranks[0:i]);

        # See if the ensemble in this setup has record-breaking validation accuracy. If so, 
        # track it.
        Results = Test_Binary_Classifier(
                        U       = Model,
                        Inputs  = Val_Inputs,
                        Targets = Val_Targets);
        
        if(Results["Accuracy"] > Max_Accuracy):
            Max_Accuracy    = Results["Accuracy"];
            Best_Num_Models = i;
        
        print("With %3u models, validation accuracy = %5.3f" % (i, Results["Accuracy"]));
    
    # Finally, set up the ensemble to recreate its best performance. 
    print("The ensemble acheives its best validation accuracy when i = %u" % Best_Num_Models);
    Model.Set_Call_Models(Sub_Model_Ranks[0:Best_Num_Models]);

    ###############################################################################################
    # Report final results. 

    print("Evaluating...");

    print("\n\t\t---===|  Training set  |===---");
    Test_Binary_Classifier_Ensemble(
            Ensemble_Model  = Model,
            Inputs          = Train_Inputs, 
            Targets         = Train_Targets,
            Call_Mode       = "average");
        
    print("\n\t\t---===| Validation set |===---");
    Test_Binary_Classifier_Ensemble(
            Ensemble_Model  = Model,
            Inputs          = Val_Inputs, 
            Targets         = Val_Targets,
            Call_Mode       = "average");

    print("\n\t\t---===|  Testing set   |===---");
    Test_Binary_Classifier_Ensemble(
            Ensemble_Model  = Model,
            Inputs          = Test_Inputs, 
            Targets         = Test_Targets,
            Call_Mode       = "average");



if __name__ == "__main__":
    main_Ensemble();
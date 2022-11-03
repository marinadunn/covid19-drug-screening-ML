import  tensorflow  as      tf;
import  numpy;
from    typing      import  List, Dict, Tuple; 
import  time;

from    Test_Train  import Train, Test, Test_Binary_Classifier;
from    Network     import Network;
from    Ensemble    import Ensemble;
from    Loss        import L2_Loss



def Train_Ensemble( Train_Inputs    : tf.Tensor, 
                    Train_Targets   : tf.Tensor, 
                    Val_Inputs      : tf.Tensor, 
                    Val_Targets     : tf.Tensor,
                    Ensemble_Model  : Ensemble,
                    Num_Epochs      : int, 
                    Optimizer       : tf.keras.optimizers.Optimizer,
                    Normalizer      : str       = "none",
                    Lambda          : float     = 0.0001) -> None:
    """ 
    This function trains an ensemble of Models, each one on a different subset of features. It 
    first generates a feature subset for each model. It then trains each model on its feature 
    subset. After calling this function, the Models should be trained. 

    -----------------------------------------------------------------------------------------------
    Arguments: 
    
    Train_Inputs: The inputs for the training set. This should be a 2D tensor. Each row corresonds 
    to an input and each column correspons to a feature. Thus, the number of features is the number 
    of columns in Inputs. 

    Train_Targets: The target values for the training set. This should be a 2D tensor. Each row 
    corresponds to a target value. The number of columns should match the dimension of the 
    co-domain of each network. 

    Val_Inputs: The inputs for the validation set. This should be a 2D tensor. Each row corresonds 
    to an input and each column correspons to a feature.

    Val_Targets: The inputs for the validation set. This should be a 2D tensor. Each row 
    corresponds to a target value.

    Ensemble_Model: An ensemble object containing the sub-models we want to train. Each model 
    should be a Network object whose domain dimension is Subset_Size and whose co-domain dimension 
    is the number of columns in Targets. 

    Num_Epochs: The number of epochs we should train each model for. 

    Optimizer: The optimizer we should use to train the models. 

    Normalizer: Specifies which normalizer we should use. Options are "L1", "L2", and "none". If 
    you select L1 or L2, we will add lambda times L1 or L2 norm of U's parameters to the Loss. 

    Lambda: If "Normalizer" is "L1" or "L2", then we add lambda times the L1 or L2 norm of U's 
    parameters to the Loss. Otherwise, we ignore this argument.

    -----------------------------------------------------------------------------------------------
    Returns:

    Nothing!
    """

    # Make sure the Inputs/Targets have the right shape.
    assert(len(Train_Inputs.shape)  == 2);   
    assert(len(Train_Targets.shape) == 2);     

    assert(len(Val_Inputs.shape)   == 2);   
    assert(len(Val_Targets.shape)  == 2); 

    assert(Train_Inputs.shape[0]    == Train_Targets.shape[0]);
    assert(Val_Inputs.shape[0]      == Val_Targets.shape[0]);

    assert(Train_Inputs.shape[1]    == Ensemble_Model.Dim_In);
    assert(Train_Targets.shape[1]   == Ensemble_Model.Dim_Out);


    # Train the models! 
    Num_Models  : int   = Ensemble_Model.Num_Sub_Models;
    Time_Start  : float = time.time();
    print("Training an ensemble with %u sub-models." % Num_Models);

    for m in range(Num_Models):
        print("Training sub-model %3u/%u" % (m + 1, Num_Models));
        
        # Get the ith model.
        Model : Network = Ensemble_Model.Sub_Models[m];

        """# Report which features it uses.
        print("Features - ", end = '');
        Sorted_Features : List = tf.sort(Model.Feature_Subset).numpy().tolist();
        print(Sorted_Features);"""

        # Train the ith model; keep track of when the training loss is at a minimum.
        Best_Loss       : float     = 9.99;
        Best_Epoch      : int       = 0;
        Best_Model      : Network   = Model.Copy(); 

        for i in range(Num_Epochs):
            # Print epoch number and progress bar
            print("%4u/%u epochs complete - " % (i, Num_Epochs), end = '');
            Num_Bars        : int = int(50*(i / Num_Epochs));
            Progress_Bar    : str = "[" + Num_Bars*"#" + (50 - Num_Bars)*" " + "] ";
            print(Progress_Bar, end = '');

            # Train!
            Train(  U           = Model, 
                    Inputs      = Train_Inputs, 
                    Targets     = Train_Targets, 
                    Optimizer   = Optimizer,
                    Normalizer  = Normalizer,
                    Lambda      = Lambda);
            
            # Test!
            Test_Loss : float = Test(   U       = Model, 
                                        Inputs  = Val_Inputs, 
                                        Targets = Val_Targets);
            
            # Update the best loss if current loss beats the previous best. 
            if(Test_Loss < Best_Loss):
                Best_Loss       = Test_Loss;
                Best_Epoch      = i;
                Best_Model      = Model.Copy();


            # Report best epoch.
            print("Lowest loss = %7.4f on epoch %4u." % (Best_Loss, Best_Epoch + 1), end = '\r');


        # Print final epoch number and progress bar
        print("%4u/%u epochs complete - " % (Num_Epochs, Num_Epochs), end = '');
        Progress_Bar    : str = "[" + 50*"#" + "] ";
        print(Progress_Bar, end = '');
        print("Lowest loss = %7.4f on epoch %4u." % (Best_Loss, Best_Epoch + 1));

        # Replace the Model with its best version.
        Model                           = Best_Model;
        Ensemble_Model.Sub_Models[m]    = Best_Model;

        # Test the final model.
        Val_Results : Dict = Test_Binary_Classifier(   
                                    U       = Model, 
                                    Inputs  = Val_Inputs, 
                                    Targets = Val_Targets);

        # Report validation results.
        print("Validation set:");
        print("Class 0 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   % (Val_Results["Precision 0"],  Val_Results["Recall 0"],    Val_Results["F1 0"]));
        print("Class 1 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   % (Val_Results["Precision 1"],  Val_Results["Precision 1"], Val_Results["F1 1"]));
        print("        | Accuracy  = %6.3f, Data Loss   = %6.3f, L2 Loss = %6.3f\n" % (Val_Results["Accuracy"],     Val_Results["Data Loss"],   L2_Loss(Model)));
        print();
    
    # Calculate final time. 
    Time_Final  : float = time.time();
    Runtime     : float = Time_Final - Time_Start;
    print("Done! took %fs. That's an average of %fs per model (%fs per epoch)" 
            % ( Runtime, 
                Runtime/Num_Models, 
                Runtime/(Num_Models*Num_Epochs)));



def Test_Binary_Classifier_Ensemble( 
            Ensemble_Model  : Ensemble,
            Inputs          : tf.Tensor,
            Targets         : tf.Tensor,
            Batch_Size      : int       = 64,
            Call_Mode       : str       = "both") -> None:
    """ 
    This function performs testing for an ensemble of classifier.

    -----------------------------------------------------------------------------------------------
    Arguments: 

    Ensemble_Model: An Ensemble object whose sub-models are binary classifiers. 

    Inputs: The inputs that U should map to the Targets. This is a 2D tensor whose ith row holds 
    the ith input.

    Targets: The targets that U should map the Inputs to. This is a 2D tensor, whose ith row holds
    the ith target value.

    Batch_Size: We evaluate the loss in mini-batches. This is the size of each mini-batch.

    Call_Mode: This determines which evaluation mode we use when evaluating the Ensemble. Options
    are "both", "averaging", and "voting". This argument is case-insensitive. 

    -----------------------------------------------------------------------------------------------
    Returns: 

    Nothing!
    """

    if  (Call_Mode.lower()[0] == "b"):
        Mode = "both";
    elif(Call_Mode.lower()[0] == "a"):
        Mode = "averaging";
    elif(Call_Mode.lower()[0] == "v"):
        Mode = "voting";
    
    # Evaluate using "vote" mode.
    if(Mode == "voting" or Mode == "both"):
        Ensemble_Model.Set_Call_Mode("vote");
        print("Voting:      ");

        Results : Dict = Test_Binary_Classifier( 
                                U           = Ensemble_Model, 
                                Inputs      = Inputs, 
                                Targets     = Targets,
                                Batch_Size  = Batch_Size);
        
        print("Class 0 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   %  (Results["Precision 0"], Results["Recall 0"], Results["F1 0"]));
        print("Class 1 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   % (Results["Precision 1"], Results["Precision 1"], Results["F1 1"]));
        print("        | Accuracy  = %6.3f, Data Loss   = %6.3f" %   (Results["Accuracy"],  Results["Data Loss"]));
        print();

    # Evaluate using "average" mode.
    if(Mode == "averaging" or Mode == "both"):
        Ensemble_Model.Set_Call_Mode("average");
        print("Averaging:   ");

        Results : Dict = Test_Binary_Classifier( 
                                U           = Ensemble_Model, 
                                Inputs      = Inputs, 
                                Targets     = Targets,
                                Batch_Size  = Batch_Size); 
        
        print("Class 0 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   %  (Results["Precision 0"], Results["Recall 0"], Results["F1 0"]));
        print("Class 1 | Precision = %6.3f, Recall      = %6.3f, F1      = %6.3f"   % (Results["Precision 1"], Results["Precision 1"], Results["F1 1"]));
        print("        | Accuracy  = %6.3f, Data Loss   = %6.3f" %   (Results["Accuracy"],  Results["Data Loss"]));
        print();

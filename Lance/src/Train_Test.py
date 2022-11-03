import  numpy;
import  torch;
from    typing      import  Dict, List;

from    GNN         import  GNN;
from    Loss        import  L2_Squared;
from    Loss        import  SSE_Loss    as Data_Loss;




def Train(  Model       : GNN, 
            Optimizer   : torch.optim.Optimizer,
            Loader,
            Lambda      : Dict) -> None:
    """
    This function performs one epoch of training on the Model using the passed optimizer.

    -----------------------------------------------------------------------------------------------
    Arguments: 

    Model: This is a GNN model, it is what we train. 

    Optimizer: The optimizer we use to train the Model. It must have been initialized using Model's
    parameters.

    Loader: A loader object which holds the data we want to use to train the Model.

    Lambda: A Dictionary containing the weights for the various terms in the loss function used to
    train Lambda. In particular, it should contain "Data" and "L2" keys, whose corresponding 
    values should be the weight of the Data Loss and L2 Loss, respectively in the loss function:
        Loss(Theta) = Lambda["Data"]*Data_Loss(ModelTheta), Targets) 
                    + Lambda["L2"]*L2_Squared(Model(Theta))
    """

    # Put the model into training mode!
    Model.train();

    # Loop through the batches.
    for Data in Loader:
        def Closure() -> torch.Tensor:
            # Zero the gradients of the Model's tensors.
            Optimizer.zero_grad();

            # Pass the data through the network, store the output on our device.
            Predictions : torch.Tensor = Model(Data).reshape(-1);
        
            # Evaluate the loss.
            Loss = (Data_Loss(Predictions, Data.y)*Lambda["Data"] +
                    L2_Squared(Model)*Lambda["L2"]);

            # Run back-propigation.
            Loss.backward();

            # Return Loss.
            return Loss;

        # Update the network models.
        Optimizer.step(Closure);



def Test(   Model       : GNN, 
            Loader,
            Lambda      : Dict) -> Dict:
    """
    This function evaluates the Model's performance on the dataset contained in the Loader. It 
    returns its findings in a dictionary.

    -----------------------------------------------------------------------------------------------
    Arguments:

    Model: A GNN model. We evaluate the performance of this model.

    Loader: A data loader that holds the data set opon which we want to evaluate the Model.

    Lambda: A dictionary containing the weights for the various terms in the loss function. 
    In particular, this dictionary should contain two keys, "Data" and "L2", whose corresponding
    values hold the weight of the Data loss and L2 loss in the loss function:
        Loss(Theta) = Lambda["Data"]*Data_Loss(ModelTheta), Targets) 
                    + Lambda["L2"]*Mean_Squared_Parameter(Model(Theta))
    """

    # First, put the model in evaluation mode. 
    Model.eval();

    # Initialize stuff to track how the Model classifies the graphs.
    Num_Data        : int   = 0;

    True_Positives  : int   = 0;
    False_Negatives : int   = 0;
    True_Negatives  : int   = 0;
    False_Positives : int   = 0;

    Total_Data_Loss : float = 0;

    # Loop through the batches.
    with torch.no_grad():
        for Data in Loader:
            # Pass the data through the Model to get predictions.
            Pred = Model(Data).reshape(-1);

            # Update number of data points.
            Num_Data += torch.numel(Pred);

            # Evaluate the data Loss.
            Total_Data_Loss += Data_Loss(Pred, Data.y);

            # Round the predictions.
            Rounded_Predictions : torch.Tensor = torch.round(Pred).to(torch.int32);

            # Cast y to int32.
            y : torch.Tensor = Data.y.to(torch.int32);

            # Determine which predictions are correct!
            Correct_Predictions : torch.Tensor = torch.eq(Rounded_Predictions, y);

            # Count the number of true/false positives/negatives
            True_Positives  += torch.sum(torch.logical_and(Correct_Predictions,                     y)); 
            False_Positives += torch.sum(torch.logical_and(torch.logical_not(Correct_Predictions),  torch.logical_not(y)));
            True_Negatives  += torch.sum(torch.logical_and(Correct_Predictions,                     torch.logical_not(y)));
            False_Negatives += torch.sum(torch.logical_and(torch.logical_not(Correct_Predictions),  y));


    # Calculate mean data loss.
    Mean_Data_Loss  : float = Total_Data_Loss / Num_Data;
    L2_Loss         : float = L2_Squared(Model).item();
    Total_Loss      : float = (Mean_Data_Loss*Loader.batch_size)*Lambda["Data"] + L2_Loss*Lambda["L2"];

    # Store the results in a dictionary.
    Results_Dict = {"True Positives"    : True_Positives, 
                    "False Negatives"   : False_Negatives,
                    "True Negatives"    : True_Negatives,
                    "False Positives"   : False_Positives,
                    "Mean Data Loss"    : Mean_Data_Loss,
                    "L2 Loss"           : L2_Loss,
                    "Total Loss"        : Total_Loss};

    # All done, return!
    return Results_Dict;



def Report_Test_Results(Test_Results : Dict) -> None:
    """
    This function merely reports the results returned by the Test function.

    -----------------------------------------------------------------------------------------------
    Arguments:

    Test_Results: This is a dictionary. It should be the dictionary returned by Test.
    """

    # Calculate actual/predicted positives/negatives and number correct.
    Actual_Positives    : int = Test_Results["True Positives"] + Test_Results["False Negatives"];
    Actual_Negatives    : int = Test_Results["True Negatives"] + Test_Results["False Positives"];
    Predicted_Positives : int = Test_Results["True Positives"] + Test_Results["False Positives"];
    Predicted_Negatives : int = Test_Results["True Negatives"] + Test_Results["False Negatives"];

    Correct             : int = Test_Results["True Positives"] + Test_Results["True Negatives"];

    # Calculate true positive/negative ratio, and positive/negative positive value.
    TPR         : float = Test_Results["True Positives"] / Actual_Positives;
    PPV         : float = Test_Results["True Positives"] / Predicted_Positives;
    TNR         : float = Test_Results["True Negatives"] / Actual_Negatives;
    NPV         : float = Test_Results["True Negatives"] / Predicted_Negatives;

    # Calculate F1, Accuracy.
    F1          : float = 2*(PPV * TPR) / (PPV + TPR);
    Accuracy    : float = Correct / (Actual_Positives + Actual_Negatives);

    # First, report classification statistics.
    print("Accuracy  = %8.6f | F1      = %8.6f" % (Accuracy, F1));
    print("TPR       = %8.6f | PPV     = %8.6f" % (TPR, PPV));
    print("TNR       = %8.6f | NPV     = %8.6f" % (TNR, NPV));
    
    # Next, report the losses.
    print(  "Data Loss = %8.2e | L2 Loss = %8.2e | Total = %8.2e" % 
            (Test_Results["Mean Data Loss"], Test_Results["L2 Loss"], Test_Results["Total Loss"]),
             end = "\n\n");
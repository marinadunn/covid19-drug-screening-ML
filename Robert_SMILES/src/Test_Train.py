import  tensorflow  as      tf;
import  numpy;
from    typing      import  Dict, Callable, List;

from    Network     import  Network;
from    Loss        import  SSE_Loss    as  Data_Loss;
from    Loss        import  L1_Loss, L2_Loss;



def Train(  U           : tf.keras.Model, 
            Inputs      : tf.Tensor, 
            Targets     : tf.Tensor, 
            Optimizer   : tf.keras.optimizers.Optimizer,
            Batch_Size  : int       = 64,
            Normalizer  : str       = "none",
            Lambda      : float     = 0.0001) -> None:
    """ 
    This function performs one epoch of training. 

    -----------------------------------------------------------------------------------------------
    Arguments: 

    U: A neural network object. 

    Inputs: The inputs that U should map to the Targets. This is a 2D tensor whose ith row holds 
    the ith input.

    Targets: The targets that U should map the Inputs to. This is a 2D tensor, whose ith row holds
    the ith target value.

    Optimizer: An Optimizer object we use to update the network weights/biases.

    Batch_Size: We perform training in mini-batches. This is the size of each mini-batch.

    Normalizer: Specifies which normalizer we should use. Options are "L1", "L2", and "none". If 
    you select L1 or L2, we will add lambda times L1 or L2 norm of U's parameters to the Loss. 

    Lambda: If "Normalizer" is "L1" or "L2", then we add lambda times the L1 or L2 norm of U's 
    parameters to the Loss. Otherwise, we ignore this argument.

    -----------------------------------------------------------------------------------------------
    Returns: 

    Nothing! 
    """

    # Fetch number of training examples. 
    N : int = Inputs.shape[0];

    # Cycle through the batches (main loop).
    for i in range(0, N - Batch_Size, Batch_Size):
        # Fetch inputs/targets for this batch.
        Inputs_Batch  : tf.Tensor = Inputs[ i:(i + Batch_Size), :];
        Targets_Batch : tf.Tensor = Targets[i:(i + Batch_Size), :];

        # Start a GradientTape instance for this batch. 
        with tf.GradientTape() as tape:
            # Get SSE loss. 
            loss : tf.Tensor = Data_Loss(
                                    U           = U, 
                                    Inputs      = Inputs_Batch, 
                                    Targets     = Targets_Batch, 
                                    Training    = True);
            
            # If we are using L1 or L2 norm, add that in.
            if  (Normalizer == "L1"):
                loss += tf.scalar_mul(Lambda, L1_Loss(U));
            elif(Normalizer == "L2"):
                loss += tf.scalar_mul(Lambda, L2_Loss(U));

            # Compute the gradient of loss with respect to the model weights.
            Grads : tf.Tensor = tape.gradient(
                                        target  = loss, 
                                        sources = U.trainable_weights);
    
            # Run the optimizer!
            Optimizer.apply_gradients(zip(Grads, U.trainable_weights));

    # Final batch (clean up). First, fetch inputs/targets for this batch.
    Inputs_Batch  : tf.Tensor = Inputs[ (i + Batch_Size):, :];
    Targets_Batch : tf.Tensor = Targets[(i + Batch_Size):, :];

    # Start a GradientTape instance for this batch. 
    with tf.GradientTape() as tape:
        # Get SSE loss. 
        loss : tf.Tensor = Data_Loss(
                                U           = U, 
                                Inputs      = Inputs_Batch, 
                                Targets     = Targets_Batch, 
                                Training    = True);

        # If we are using L1 or L2 norm, add that in.
        if  (Normalizer == "L1"):
            loss += tf.scalar_mul(Lambda, L1_Loss(U));
        elif(Normalizer == "L2"):
            loss += tf.scalar_mul(Lambda, L2_Loss(U));

        # Compute the gradient of loss with respect to the model weights.
        Grads : tf.Tensor = tape.gradient(
                                    target  = loss, 
                                    sources = U.trainable_weights);

        # Run the optimizer!
        Optimizer.apply_gradients(zip(Grads, U.trainable_weights));



def Test(   U           : tf.keras.Model, 
            Inputs      : tf.Tensor, 
            Targets     : tf.Tensor, 
            Batch_Size  : int       = 64) -> float: 
    """ 
    This function performs testing (average, per component loss between inputs and targets).

    -----------------------------------------------------------------------------------------------
    Arguments: 

    U: A neural keras Model object, specifically a binary classifier. 

    Inputs: The inputs that U should map to the Targets. This is a 2D tensor whose ith row holds 
    the ith input.

    Targets: The targets that U should map the Inputs to. This is a 2D tensor, whose ith row holds
    the ith target value.

    Batch_Size: We evaluate the loss in mini-batches. This is the size of each mini-batch.

    -----------------------------------------------------------------------------------------------
    Returns: 

    The average loss per component (a float) between U's predictions and the Targets.
    """

    # First, make sure that Inputs/Targets have the right shape.
    assert(len(Inputs.shape)    == 2);
    assert(len(Targets.shape)   == 2);
    assert(Inputs.shape[0]      == Targets.shape[0]);

    # Fetch number of examples.
    Num_Examples            : int = Inputs.shape[0];
    Num_Target_Components   : int = Targets.shape[1];

    # Initialize loss.
    Total_Loss : float = 0;

    # cycle through mini-batches (main loop)
    for i in range(0, Num_Examples - Batch_Size, Batch_Size):
        # Get this batch of inputs, targets.
        Inputs_Batch    : tf.Tensor = Inputs[ i:(i + Batch_Size), :];
        Targets_Batch   : tf.Tensor = Targets[i:(i + Batch_Size), :];

        # Accumulate loss.
        Total_Loss += Data_Loss(U           = U,
                                Inputs      = Inputs_Batch,
                                Targets     = Targets_Batch).numpy().item();

    # Final batch (clean up).
    Inputs_Batch    : tf.Tensor = Inputs[ (i + Batch_Size):, :];
    Targets_Batch   : tf.Tensor = Targets[(i + Batch_Size):, :];

    # Accumulate loss.
    Total_Loss += Data_Loss(U           = U,
                            Inputs      = Inputs_Batch,
                            Targets     = Targets_Batch).numpy().item();
    
    # Return 
    return Total_Loss / (Num_Examples*Num_Target_Components);




def Test_Binary_Classifier(   
            U           : tf.keras.Model, 
            Inputs      : tf.Tensor, 
            Targets     : tf.Tensor, 
            Batch_Size  : int       = 64) -> Dict[str, float]: 
    """ 
    This function performs testing for a binary classifier and returns a dictionary housing the
    results.

    -----------------------------------------------------------------------------------------------
    Arguments: 

    U: A neural keras Model object, specifically a binary classifier. 

    Inputs: The inputs that U should map to the Targets. This is a 2D tensor whose ith row holds 
    the ith input.

    Targets: The targets that U should map the Inputs to. This is a 2D tensor, whose ith row holds
    the ith target value.

    Batch_Size: We evaluate the loss in mini-batches. This is the size of each mini-batch.

    -----------------------------------------------------------------------------------------------
    Returns: 

    A dictionary housing the results of testing the model. 
    """

    # Fetch number of examples.
    Num_Examples : int = Inputs.shape[0];

    # Initialize loss.
    Total_Loss : float = 0;

    # Initialize an array to hold the predicted classes.
    Predicted_Classes : tf.Tensor = numpy.empty(shape = (Num_Examples, 1), dtype = numpy.bool);

    # cycle through mini-batches (main loop)
    for i in range(0, Num_Examples - Batch_Size, Batch_Size):
        # Get this batch of inputs, targets.
        Inputs_Batch    : tf.Tensor = Inputs[ i:(i + Batch_Size), :];
        Targets_Batch   : tf.Tensor = Targets[i:(i + Batch_Size), :];

        # Accumulate loss.
        Total_Loss += Data_Loss(U           = U,
                                Inputs      = Inputs_Batch,
                                Targets     = Targets_Batch).numpy().item();
        
        # Get predictions.
        Predict_Batch : tf.Tensor = U(Inputs_Batch);

        # Determine which class the network predicts (if output is < .5, it predicts class 1, 
        # otherwise it predicts class 2)
        Predicted_Classes[i:(i + Batch_Size), :] = tf.greater(Predict_Batch, 0.5).numpy();

    # Final batch (clean up).
    Inputs_Batch    : tf.Tensor = Inputs[ (i + Batch_Size):, :];
    Targets_Batch   : tf.Tensor = Targets[(i + Batch_Size):, :];

    # Accumulate loss.
    Total_Loss += Data_Loss(U           = U,
                            Inputs      = Inputs_Batch,
                            Targets     = Targets_Batch).numpy().item();

    # Get predictions.
    Predict_Batch : tf.Tensor = U(Inputs_Batch);

    # Determine which class the network predicts (if output is < .5, it predicts class 1, 
    # otherwise it predicts class 2)
    Predicted_Classes[(i + Batch_Size):, :] = tf.greater(Predict_Batch, 0.5).numpy();

    # Determine number of true positives.
    Num_T1 : int = numpy.sum(numpy.logical_and(Targets, Predicted_Classes));
    Num_T0 : int = numpy.sum(numpy.logical_and(numpy.logical_not(Targets), numpy.logical_not(Predicted_Classes)));
    Num_F1 : int = numpy.sum(numpy.logical_and(numpy.logical_not(Targets), Predicted_Classes)); 
    Num_F0 : int = numpy.sum(numpy.logical_and(Targets, numpy.logical_not(Predicted_Classes)));

    # Determine average loss, accuracy, precision, and recall. 
    Results : Dict[str, float] = {};

    Results["Data Loss"]    : float = Batch_Size*(Total_Loss / float(Num_Examples));
    Results["Accuracy"]     : float = float(Num_T1 + Num_T0)/float(Num_Examples);

    Results["Precision 0"]  : float = float(Num_T0) / float(max(Num_T0 + Num_F0, 1));
    Results["Recall 0"]     : float = float(Num_T0) / float(max(Num_T0 + Num_F1, 1)); 

    Results["Precision 1"]  : float = float(Num_T1) / float(max(Num_T1 + Num_F1, 1));
    Results["Recall 1"]     : float = float(Num_T1) / float(max(Num_T1 + Num_F0, 1)); 

    try:
        Results["F1 0"]     : float = 2./(1./Results["Precision 0"] + 1./Results["Recall 0"]);
    except ZeroDivisionError:
        Results["F1 0"]     : float = -1;

    try:
        Results["F1 1"]     : float = 2./(1./Results["Precision 1"] + 1./Results["Recall 1"]);
    except ZeroDivisionError:
        Results["F1 1"]     : float = -1;

    return Results;




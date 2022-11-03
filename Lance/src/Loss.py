import  torch;



def L2_Squared(Model : torch.nn.Module) -> torch.Tensor:
    """
    This function calculates the square of the L2 norm of the Model's parameter vector.
    suppose the Model is paramaterized by the vector Theta \in \mathbb{R}^P. This function 
    evaluates and returns
        \sum_{i = 1}^{P} Theta_i^2. 
    This function should work on any torch.nn.Module object. 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Model: A torch.nn.Model object which is paramterized by the vector Theta \in \mathbb{R}^P.

    -----------------------------------------------------------------------------------------------
    Returns:

    A single element tensor whose lone element holds the value \sum_{i = 1}^{P} Theta_i^2.
    """

    # First, let's get the Model's parameters.
    Parameter_List = Model.parameters();

    # Next, let's initialize buffers to track the sum of the squares of Model's parameters. 
    Squared_Sum : torch.Tensor = torch.zeros(1, dtype = torch.float32, device = next(Model.parameters()).device);

    for W in Parameter_List:
        # Add the sum of the squares of the components of W to the Squared Sum.
        Squared_Sum += torch.sum(torch.multiply(W, W));

    # Now, return the Squared Sum.
    return Squared_Sum;



def SSE_Loss(Predictions : torch.Tensor, Targets : torch.Tensor) -> torch.Tensor:
    """
    This function returns the sum of the squares of the components of the difference of Predictions
    and Targets. That is, if R = Predictions - Targets, then this function returns
        \sum_{i = 1}^{N} R_i^2 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Predictions: A set of predictions. This should be a N element tensor, hopefully output by a
    torch.nn.Module object.

    Targets: This is a set of target values. This should also be an N element tensor.

    -----------------------------------------------------------------------------------------------
    Returns:

    The sum of the squares of the components of Predictions - Targets. See above.
    """


    # First, calculate the residual (difference between the predictions and targets)
    Residual : torch.Tensor = torch.subtract(Predictions, Targets);

    # Now return the sum of the squares of the residuals.
    return torch.sum(torch.multiply(Residual, Residual));



BCE_Loss = torch.nn.BCELoss(reduction = "sum");
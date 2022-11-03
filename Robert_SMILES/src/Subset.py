import  numpy; 
import  random;
from    typing      import  List;



def Generate_Index_Subsets( Superset_Size   : int,
                            Subset_Size     : int, 
                            Num_Subsets     : int) -> numpy.ndarray:
    """ 
    This function is designed to generate the indices of random subsets of an indexed set with a 
    particular size. Specifically, it generates a specified number of subsets of a particular size 
    from the set {0, 1, 2, ... , N - 1}, where N = Superset_Size. For example, if 
    Superset_Size = 100, Subset_Size = 10, and Num_Subsets = 2, this function may return the array 
            [[43, 59, 23, 23, 96, 34, 49, 43, 69, 35], 
             [34, 81, 32,  3, 56, 82, 13,  5, 60, 98]].
    This function packages the subsets together in a 2D array, with one subset per row. Thus, the
    returned array has shape Num_Subsets x Subset_Size. 

    -----------------------------------------------------------------------------------------------
    Arguments:

    Superset_Size: The number of elements in the superset that we want to sample. If 
    Superset_Size = N, then this function yields subsets from {0, 1, ... , N - 1}.

    Subset_Size: The number of elements in each subset. Can not be greater than Superset_Size. 

    Num_Subsets: The number of subsets we generate. 

    -----------------------------------------------------------------------------------------------
    Returns:

    A 2D numpy array whose ith row contains the indices of the ith subset. This array has shape 
    Num_Subsets x Subset_Size. 
    """

    # First, make sure the inputs are valid.
    assert(Superset_Size    > 0);
    assert(Subset_Size      > 0);
    assert(Num_Subsets      > 0);
    assert(Superset_Size    > Subset_Size);

    # Initialize array to hold the subsets. 
    Subsets : numpy.ndarray = numpy.empty(shape = (Num_Subsets, Subset_Size), dtype = numpy.int32);

    # Generate the subsets!
    for i in range(Num_Subsets):
        Subsets[i, :] = random.sample(range(Superset_Size), k = Subset_Size);

    # All done!
    return Subsets;
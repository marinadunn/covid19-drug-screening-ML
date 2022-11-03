import  numpy;
import  math;
from    typing  import List;



def main(   File_Name   : str   = "mpro_exp_data2_rdkit_feat.csv", 
            Verbose     : bool  = True) -> None:
    """ This function reads in the raw data from the file specified above, eliminates all 
    problematic entries, and then uses the resulting "power washed" data to build the testing, 
    training, and validation sets. 
    
    The "File_Name" variable above specifies the file we read from. This should be the file 
    for part 1 of the challenge problem. This code reads in each line of that file, and then 
    parses the contents of each line. In particular, we determine the target value (bind or 
    no bind), input (features), and set (train, test, or validation) information of each line. 
    Some lines are missing entries. We discard these lines. After processing the data, we 
    calculate the mean and standard deviation of each feature. Then, for each row, for each j, 
    we evaluate the number of standard deviations (of the jth feature) between the mean of 
    the jth feature and the value of the jth feature in that row. We sum the resulting values 
    for each row, then rank the rows according to this "row deviation" score. We discard any 
    rows whose row deviation is too large (indiciating an outlier. We define "too large" below).
    After that, we generate the testing/training/validation input/target sets, then save those 
    sets to a .npz file. 
    

    
    -----------------------------------------------------------------------------------------------
    Inputs: 
    
    File_Name: The name of the file that contains the raw, unprocessed data. We assume the 
    following about this file:
        - 1st line contains the column headers
        - 1st column specifies example number
        - 2nd column is blank
        - 3rd column specifies the compound id corresponding to each row.
        - 4th column specifies the SMILEs encoding of each row's molecule
        - 5th column specifies if each row binds (1) or does not bind (0)
        - 6th column specifies which set (testing, training, validation) each row belongs to.
        - The next 208 rows specify the 208 features. 

    Verbose: A boolean which specifies if this function should print extra information while it
    runs or not. 
    


    -----------------------------------------------------------------------------------------------
    Returns: 

    Nothing... though we do write the testing/training/validation input/training sets to a file 
    called "Cleaned_data.npz". """

    Num_Cols    : int   = 208;



    ###############################################################################################
    # Read in data.

    print("Reading data...");

    # First, open the file. 
    Data_File = open("./" + File_Name);

    # Read the file line-by-line. Toss the first line (which contains the column headers)
    Lines : List[str] = Data_File.readlines()[1:];

    # Get the number of lines. We will need this number throughout this function.
    Num_Lines : int = len(Lines);

    # We now need to remove lines that are missing data. 
    # To do this, we split lines using the separator ',,'. Each line should split into at least 
    # two pieces (as the start of the ith line should read "i,,"). If this split yields more than 2
    # pieces, then we should remove that line. 
    Num_Removed : int = 0;
    i           : int = 0;
    while(True):
        if(i >= Num_Lines):
            break;

        # Get the ith line. 
        Line : str = Lines[i];

        # Check if line is missing entries.
        if(len(Line.split(',,')) > 2):
            # Pop the line. 
            Lines.pop(i);

            # Update i, Num_Lines (we need to decrement i since the former i+1th line is now the ith one)
            i           -= 1;
            Num_Lines   -= 1;
            Num_Removed += 1;

            # Let the user know what we're doing.            
            if(Verbose):
                print("Line %u was missing data... removing!" % (i + Num_Removed));

        # Increment i.
        i += 1;

    # Print final number of lines
    print("Removed %d lines" % Num_Removed);
    print("Number of lines after removing incomplete lines = %u" % Num_Lines); 

    # Set up arrays that keep track of the subset each line belongs to. 
    Is_Train : numpy.ndarray = numpy.zeros(shape = Num_Lines, dtype = numpy.byte);
    Is_Valid : numpy.ndarray = numpy.zeros(shape = Num_Lines, dtype = numpy.byte);
    Is_Test  : numpy.ndarray = numpy.zeros(shape = Num_Lines, dtype = numpy.byte);
    
    # Set up an array to hold the Inputs. 
    Inputs : numpy.ndarray = numpy.empty(shape = (Num_Lines, Num_Cols), dtype = numpy.float32);
    
    # Set up an array to hold the targets. 
    Targets : numpy.ndarray = numpy.empty(shape = (Num_Lines, 1), dtype = numpy.byte);

    # Process each line. 
    for i in range(Num_Lines):
        # Split the row into its components (this is a line of a csv, so ',' is the separator)
        ith_Line_Entries : List[str] = Lines[i].lower().split(',');

        # extract subset, input (last 208 columns), target portions of the ith line.
        Subset      : str           = ith_Line_Entries[5].strip().lower();
        Inputs[ i, :]               = numpy.array(ith_Line_Entries[6:]).astype(numpy.float32);
        Targets[i, :]               = int(ith_Line_Entries[4]);

        # Parse Subset.
        if  (Subset == "train"):
            Is_Train[i] = 1;
        elif(Subset == "valid"):
            Is_Valid[i] = 1;
        else:
            Is_Test[i] = 1;

    # Let's see what we read in. 
    if(Verbose):
        # Print out shape of the inputs array.
        print("Input Shape = ", end = '');
        print(Inputs.shape);

        # Determine how many nan's Inputs containts. 
        print("Number of nan: %u" % numpy.sum(numpy.isnan(Inputs)) );
        
        # Determine how many inf's Inputs contains.
        print("Number of inf: %u" % numpy.sum(numpy.isinf(Inputs)));

    # We are done reading data... close the file. 
    Data_File.close();



    ###############################################################################################
    # Outlier detection. 

    print("\nDetecting outliers...");

    # Now let's perform an outliner search. First, let's compute the mean and SD of each column.
    Col_Means   : numpy.ndarray = numpy.mean(Inputs, axis = 0);
    Col_STDs    : numpy.ndarray = numpy.std(Inputs, axis = 0);

    # Set up an array to hold each rows "row deviation". Here, I define a row's "row deviation"
    # as the sum over j of the distance (in standard deviations of the jth feature) from the 
    # jth entry of the row and the mean of the jth feature . If a row has a large row deviation, 
    # then many of its entries are outliers in their respective columns.  such a row may be a 
    # result of a bad simulation run and should probably be discarded.   
    Row_Deviation : numpy.ndarray = numpy.zeros(shape = Num_Lines, dtype = numpy.float32);

    # We will need to know how many training/validation/testing pairs we remove. 
    Num_Train_Outliers : int = 0;
    Num_Valid_Outliers : int = 0;
    Num_Test_Outliers  : int = 0;

    # Now, let's search for problematic rows. We only search through the first 104 columns.
    # The first 104 columns contain floating point entries. The other columns contain integers
    # most entries in those columns are zero, with occasional exceptions. I do not want to 
    # classify these exceptions as outliers (I think they contain useful information). Thus, I 
    # only search through the floating point columns. I don't claim my approach is totally 
    # principled. 
    eps     : float = 1e-5;
    max_col : int   = 104;
    for j in range(0, max_col):
        # Calculate the number of stds from the entries in the jth column and the column's mean.
        if(Col_STDs[j] > eps):
            Row_Deviation += numpy.abs(numpy.divide(Inputs[:, j] - Col_Means[j], Col_STDs[j]));
    
    if(Verbose):
        print("the top 10 row deviations are: ", end = '')
        print(numpy.sort(Row_Deviation)[-10:]);

        print("The maximum row deviation is Row_Deviation[%d] = %f" % (numpy.argmax(Row_Deviation), numpy.max(Row_Deviation)));

    # Check for outlier rows. I somewhat arbitrarily say a row is an outlier if its row deviation 
    # is more than 3*max_col.        
    Is_Outlier : numpy.ndarray = numpy.greater(Row_Deviation, float(3*max_col));
    print("Detected %u outlier rows. " % numpy.sum(Is_Outlier));

    # Count how many outliers are in each set.
    for i in range(Num_Lines):
        if(Is_Outlier[i] == True):
            if(Is_Train[i] == True):
                Num_Train_Outliers += 1;
            elif(Is_Valid[i] == True):
                Num_Valid_Outliers += 1;
            else:
                Num_Test_Outliers  += 1;



   ###############################################################################################
    # Remove "dead" columns. 
    # Some columns have zero std. Such features contain zero useful information. We remove these 
    # columns. Doing so reduces the dimensionality of the input space without removing any 
    # information.
    
    print("\nRemoving columns with zero SD...");

    # First, compute the std of each column. 
    Stds : numpy.ndarray = numpy.std(Inputs, axis = 0);

    # Now, keep track of which columns we should eliminate. 
    Num_Dead_Cols   : int   = 0;
    Keep_Columns    : List  = [];

    for i in range(Num_Cols):
        # If column i has zero std, do not add it to the keep list.
        if(Stds[i] == 0.0):
            if(Verbose):
                print("Feature %4u has a standard deviation of 0. I am removing it." % (i + 1));
            
            Num_Dead_Cols += 1;

        # Otherwise, add it to the keep list.
        else:
            Keep_Columns.append(i);
        
    # Report the number of dead columns.
    print("Removed %u dead columns" % Num_Dead_Cols);
    
    # Keep the columns with non-zero std.
    Inputs = Inputs[:, Keep_Columns];

    # Update number of columns.
    Num_Cols = Inputs.shape[1];



    ###############################################################################################
    # Form testing, training, and validation sets (with outliers removed)

    print("\nForming sets...");

    # Determine how many testing/training/validation pairs there are.
    Num_Train : int = numpy.sum(Is_Train) - Num_Train_Outliers;
    Num_Valid : int = numpy.sum(Is_Valid) - Num_Valid_Outliers;
    Num_Test  : int = numpy.sum(Is_Test)  - Num_Test_Outliers;

    # Set up testing/training/validation input and target sets.
    Train_Inputs  : numpy.ndarray = numpy.empty(shape = (Num_Train, Num_Cols),  dtype = numpy.float32);
    Train_Targets : numpy.ndarray = numpy.empty(shape = (Num_Train, 1),         dtype = numpy.byte);
    Valid_Inputs  : numpy.ndarray = numpy.empty(shape = (Num_Valid, Num_Cols),  dtype = numpy.float32);
    Valid_Targets : numpy.ndarray = numpy.empty(shape = (Num_Valid, 1),         dtype = numpy.byte);
    Test_Inputs   : numpy.ndarray = numpy.empty(shape = (Num_Test,  Num_Cols),  dtype = numpy.float32);
    Test_Targets  : numpy.ndarray = numpy.empty(shape = (Num_Test, 1),          dtype = numpy.byte);
    
    if(Verbose):
        print("Training Inputs shape is ", end = '');
        print(Train_Inputs.shape);

        print("Training Targets shape is ", end = '');
        print(Train_Targets.shape);

        print("Validation Inputs shape is ", end = '');
        print(Valid_Inputs.shape);

        print("Validation Targets shape is ", end = '');
        print(Valid_Targets.shape);
            
        print("Testing Inputs shape is ", end = '');
        print(Test_Inputs.shape);

        print("Testing Targets shape is ", end = '');
        print(Test_Targets.shape);

    # Populate the datasets line-by-line.
    i_Data      : int = 0;          # which row we are currently processing
    i_Train     : int = 0;          # the first un-populated row of the Train arrays.
    i_Valid     : int = 0;          # the first un-populated row of the Valid arrays.
    i_Test      : int = 0;          # the first un-populated row of the Test arrays.

    while(True):
        # Check if we've reached the end of Data.
        if(i_Data >= Num_Lines):
            break;

        # If row i_Data is not an outlier, copy the input/target to the corresponding set.
        if(Is_Outlier[i_Data] == False):
            if(Is_Train[i_Data] == True):
                # Populate the i_Train'th row of Training inputs/targets.
                Train_Inputs [i_Train, :]   = Inputs [i_Data, :];
                Train_Targets[i_Train, :]   = Targets[i_Data, :];

                # Now that the row is full, increment i_Train.
                i_Train += 1;


            elif(Is_Valid[i_Data] == True):
                # Populate the i_Valid'th row of Validation inputs/targets.
                Valid_Inputs [i_Valid, :]   = Inputs [i_Data, :];
                Valid_Targets[i_Valid, :]   = Targets[i_Data, :];               

                # Now that the row is full, increment i_Valid.
                i_Valid += 1;

            else:
                # Populate the i_Test'th row of Testing inputs/targets.
                Test_Inputs [i_Test, :]     = Inputs [i_Data, :];
                Test_Targets[i_Test, :]     = Targets[i_Data, :];               

                # Now that the row is full, increment i_Test.
                i_Test += 1;
        else:
            print("Pruning row %d." % i_Data);

        # Increment i_Data.
        i_Data += 1;
    
    if(Verbose):
        print("Final i_Train = %u (Num_Train = %u)" % (i_Train, Num_Train));
        print("Final i_Valid = %u (Num_Valid = %u)" % (i_Valid, Num_Valid));
        print("Final i_Test  = %u (Num_Test  = %u)" % (i_Test,  Num_Test ));



    ###############################################################################################
    # Save and clean up. 

    print("\nSaving sets...", end = '');

    Path : str = "./Cleaned_Data.npz";
    Save_File  = open(Path, mode = "wb");

    numpy.savez(file            = Save_File,
                Train_Inputs    = Train_Inputs,
                Train_Targets   = Train_Targets,
                Test_Inputs     = Test_Inputs,
                Test_Targets    = Test_Targets,
                Valid_Inputs    = Valid_Inputs,
                Valid_Targets   = Valid_Targets);

    print("Done!");

if __name__ == "__main__":
    main();
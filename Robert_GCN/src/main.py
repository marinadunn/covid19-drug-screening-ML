import  os
from re import S;
import  sys

# Add the data repository to the search path.
main_Path   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
data_Path   = os.path.join(main_Path, "data");
sys.path.append(data_Path);

# Now import the data extractor.
from    data_extractor  import  LigandGraphIMDataset;

# ???
os.environ["CUDA_LAUNCH_BLOCKING"] = "1";

import  numpy;
import  torch;
import  torch_geometric;
from    typing          import  List, Dict;
import  time;
from    datetime        import  datetime;

from    GNN             import  GNN;
from    Train_Test      import  Train, Test, Report_Test_Results;



def Train_Model(    Settings    : Dict,
                    Loaders     : Dict,
                    Device      : torch.device) -> GNN:
    """
    This function trains a network according to the settings. It returns the final network. 

    -----------------------------------------------------------------------------------------------
    Arguments: 
    
    Settings: A dictionary housing the desired settings. This dictionary must have the following 
    keys: 
        "Load From File"    [bool]
        "Learning Rate"     [float]
        "Number of Epochs"  [int]
        "Lambda"            [Dict]
    If we are loading from file, the dict must also have a "Load File Name" key. If we are not, 
    then the dict must contain the following keys
        "Conv Widths"           [List]
        "Conv Activation"       [str]
        "Conv Type"             [str]
        "Pooling Type"          [str]
        "Pooling Activation"    [str]
        "Linear Widths"         [List]
        "Linear Activation"     [str]
        "Output Activatio"      [str]
    which are passed to the GNN initializer.

    Loader: This is a dictionary that should contain three keys: "Train", "Test", and "Validation".
    The corresponding items should be data loaders for the train, test, and validation sets, 
    respectively.

    Device: The derivce we want to initialize/train the model on.

    -----------------------------------------------------------------------------------------------
    Returns:

    The initialized and trained GNN object.
    """
    

    ###################################################################################################
    # Initialize the GNN, Optimizer.

    if(Settings["Load From File"] == True):
        # First, let's load the saved State. We will initialize a GNN model using this.
        State : Dict = torch.load("../saves/" + Settings["Load File Name"]);

        # Let's extract the stuff we need to initialize an object (widths, activations, pooling type).
        Conv_Widths         : List  = State["Conv Widths"];
        Conv_Type           : str   = State["Conv Type"];
        Conv_Activation     : str   = State["Conv Activation"];
        Pooling_Type        : str   = State["Pooling Type"];
        Pooling_Activation  : str   = State["Pooling Activation"];
        Linear_Widths       : List  = State["Linear Widths"];
        Linear_Activation   : str   = State["Linear Activation"];
        Output_Activation   : str   = State["Output Activation"];

        # Initialize the model using the saved state.
        Model = GNN(Conv_Widths         = Conv_Widths, 
                    Linear_Widths       = Linear_Widths,
                    Conv_Type           = Conv_Type);
        
        Model.Load_State(State);

    else:
        # Initialize the model using the settings above.
        Model = GNN(Conv_Widths         = Settings["Conv Widths"], 
                    Conv_Activation     = Settings["Conv Activation"],
                    Conv_Type           = Settings["Conv Type"],
                    Pooling_Type        = Settings["Pooling Type"],
                    Pooling_Activation  = Settings["Pooling Activation"],
                    Linear_Widths       = Settings["Linear Widths"],
                    Linear_Activation   = Settings["Linear Activation"],
                    Output_Activation   = Settings["Output Activation"]);

    # Move the Model to the selected device.
    Model = Model.to(Device);

    # Select the optimizer.
    Optimizer   = torch.optim.Adam(Model.parameters(), lr = Settings["Learning Rate"]);


    ###################################################################################################
    # Loop through the epochs!

    # Set up a timer to track the average runtime per epoch. 
    Epoch_Timer : float = time.perf_counter();
    print("Running %u epochs..." % Settings["Number of Epochs"]);

    # Initialize something to track the lowest training loss so far.
    Max_Correct    : int   = 0;
    Best_Model      : GNN   = Model.Copy();  

    for i in range(Settings["Number of Epochs"]):
        # Report epoch numnber
        print("Epoch %4u / %u" % (i + 1, Settings["Number of Epochs"]));

        # First, train!
        Train(  Model       = Model,
                Optimizer   = Optimizer,
                Loader      = Loaders["Train"],
                Lambda      = Settings["Lambda"]);

        # Now, test!
        Train_Results = Test(   Model   = Model, 
                                Loader  = Loaders["Train"],
                                Lambda  = Settings["Lambda"]);
        
        Val_Results = Test(     Model   = Model, 
                                Loader  = Loaders["Validation"],
                                Lambda  = Settings["Lambda"]);

        # See if Model's validation loss is a new best. If so, track it.
        Num_Correct : int = Val_Results["True Positives"] + Val_Results["True Negatives"];
        if(Num_Correct >= Max_Correct):
            # Update the Min Val Loss.
            Max_Correct = Num_Correct;

            # Replace Best Model with a deep copy of the current model.
            Best_Model = Model.Copy();

        # Report results.
        print("Training:");
        Report_Test_Results(Train_Results);
        print("Validation:");
        Report_Test_Results(Val_Results);
    
    # Report runtime.
    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs," % Epoch_Runtime);
    print("an average of %7.2fs per epoch." % (Epoch_Runtime / Settings["Number of Epochs"]));  


    ###############################################################################################
    # Report final testing, validation, and training loss for best model.

    # Evaluate the best model.
    Train_Results = Test(   Model   = Best_Model, 
                            Loader  = Loaders["Train"],
                            Lambda  = Settings["Lambda"]);
        
    Val_Results = Test(     Model   = Best_Model, 
                            Loader  = Loaders["Validation"],
                            Lambda  = Settings["Lambda"]);

    Test_Results = Test(    Model   = Best_Model, 
                            Loader  = Loaders["Test"],
                            Lambda  = Settings["Lambda"]);

    print("\nBest model results:");
    print("Training:");
    Report_Test_Results(Train_Results);
    print("Validation:");
    Report_Test_Results(Val_Results);
    print("Testing:");
    Report_Test_Results(Test_Results);

    # All done, return the final model.
    return {"Model"                 : Best_Model, 
            "Train Results"         : Train_Results, 
            "Test Results"          : Test_Results, 
            "Validation Results"    : Val_Results};



def Setup_Loaders(  Device      : torch.device,
                    Batch_Size  : int) -> Dict:
    """
    This function loads the testing, training, and validation sets from file, maps them to the 
    appropiate device, and then packages them in torch_geometric data loaders. The function then 
    returns a dictionary with three keys: "Test", "Train", and "Validation". The corresponding 
    values are the corresponding data loaders. Note that we shuffle the training set but not the 
    test or validation sets.

    -----------------------------------------------------------------------------------------------
    Arguments: 

    Device: The device we want to load the data oneo.

    Batch_Size: An integer specifing the batch size for the three data loaders.

    -----------------------------------------------------------------------------------------------
    Returns:

    A dictionary with three keys, "Test", "Train", and "Validation". The item corresponding to the 
    "Test" key is the data loader for the test set. The same applies to the "Train" and 
    "Validation" keys.
    """

    # Get the training, testing, and validation data sets from file. If any of these sets do not 
    # exist, this will generate them from the hdf5 files.
    Train_Data  = LigandGraphIMDataset(root = '../data/train_dataset',  hdf5_file_name='postera_protease2_pos_neg_train.hdf5')
    Test_Data   = LigandGraphIMDataset(root = '../data/test_dataset',   hdf5_file_name='postera_protease2_pos_neg_test.hdf5')
    Val_Data    = LigandGraphIMDataset(root = '../data/val_dataset',    hdf5_file_name='postera_protease2_pos_neg_val.hdf5')

    # To speed up runtime, we move the inputs (x), targets (y), and node structure information 
    # (edge_index) to the desired device.
    Train_Data.data.x           = Train_Data.data.x.to(Device);
    Train_Data.data.edge_index  = Train_Data.data.edge_index.to(Device);
    Train_Data.data.y           = Train_Data.data.y.to(Device);

    Test_Data.data.x            = Test_Data.data.x.to(Device);
    Test_Data.data.edge_index   = Test_Data.data.edge_index.to(Device);
    Test_Data.data.y            = Test_Data.data.y.to(Device);

    Val_Data.data.x             = Val_Data.data.x.to(Device);
    Val_Data.data.edge_index    = Val_Data.data.edge_index.to(Device);
    Val_Data.data.y             = Val_Data.data.y.to(Device);
    
    # Load the data into data loader objects. 
    Train_Loader    = torch_geometric.loader.DataLoader(   
                            Train_Data,
                            batch_size      = Batch_Size, 
                            shuffle         = True);

    Test_Loader     = torch_geometric.loader.DataLoader(   
                            Test_Data, 
                            batch_size      = Batch_Size, 
                            shuffle         = False);

    Val_Loader      = torch_geometric.loader.DataLoader(   
                            Val_Data, 
                            batch_size      = Batch_Size, 
                            shuffle         = False);
    
    # Package everything together and return.
    Loaders : Dict = {  "Train"         : Train_Loader,
                        "Validation"    : Val_Loader,
                        "Test"          : Test_Loader};
    
    return Loaders;



if __name__ == "__main__":
    ###################################################################################################
    # Get the data loaders

    # Data settings.
    Device      : torch.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    Batch_Size  : int           = 64;

    # Load the data into a set of data loders. This dict has a loader for each set (train/test/val)
    Loaders : Dict = Setup_Loaders(Device = Device, Batch_Size = Batch_Size);


    ###################################################################################################
    # Settings.

    Settings : Dict = {};

    # Save, Load settings.
    Settings["Load From File"]      : bool      = False;
    Settings["Load File Name"]      : str       = "GraphSAGE_5_12_27";           # Ignore if not loading from file.

    # Network settings. You can ignore these if loading from file.
    Settings["Conv Widths"]         : List[int] = [19, 7, 7, 7, 7, 7];
    Settings["Conv Type"]           : str       = "GAT";
    Settings["Conv Activation"]     : str       = "elu";
    Settings["Pooling Type"]        : str       = "mean";
    Settings["Pooling Activation"]  : str       = "elu";
    Settings["Linear Widths"]       : List[int] = [35, 15, 1];
    Settings["Linear Activation"]   : str       = "elu";
    Settings["Output Activation"]   : str       = "sigmoid";

    # Optimizer settings.
    Settings["Learning Rate"]       : float     = 0.01;

    # Training settings.
    Settings["Number of Epochs"]    : int       = 50;
    Settings["Lambda"]              : Dict      = {"Data" : 1.0, "L2" : 0.003};
    

    ####################################################################################################
    # Train the model!

    Results = Train_Model(Settings = Settings, Loaders = Loaders, Device = Device);


    ##############################################################################################
    # Save the model.

    # First, get the name of the file we're going to save to.
    Time                    = datetime.now();
    Save_File_Name  : str   = "../saves/" + Settings["Conv Type"] + "_" + str(Time.day) + "_" + str(Time.hour) + "_" + str(Time.minute);
    
    # Now, get the Model state.
    State : Dict = Results["Model"].Get_State();

    # Save it! 
    torch.save(State, Save_File_Name);

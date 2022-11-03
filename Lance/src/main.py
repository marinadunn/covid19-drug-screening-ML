import  os
import  sys

# Add the data repository to the search path.
main_Path   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
data_Path   = os.path.join(main_Path, "data");
sys.path.append(data_Path);

# Now import the data extractor.
from    data_extractor  import  LigandGraphIMDataset;

import  numpy;
import  torch;
import  torch_geometric;
from    typing          import  List, Dict;
import  time;
from    datetime        import  datetime;

from    GNN             import  GNN;
from    Train_Test      import  Train, Test, Report_Test_Results;



def main():
    ###################################################################################################
    # Settings.

    # Save, Load settings.
    Load_From_File      : bool      = False;
    Load_File_Name      : str       = "GCN_2_19_24";           # Ignore if not loading from file.

    # Network settings. You can ignore these if loading from file.
    Conv_Widths         : List[int] = [19, 10, 10, 10, 10, 10];
    Conv_Type           : str       = "GAT";
    Conv_Activation     : str       = "elu";
    Pooling_Type        : str       = "mean";
    Pooling_Activation  : str       = "elu";
    Linear_Widths       : List[int] = [50, 32, 1];
    Linear_Activation   : str       = "elu";
    Output_Activation   : str       = "sigmoid"

    # Optimizer settings.
    Learning_Rate   : float = 0.01;

    # Training settings.
    Num_Epochs  : int   = 100;
    Lambda      : Dict  = {"Data" : 1.0, "L2" : 0.001};

    # Data settings.
    Batch_Size    : int = 64;
    

    ###################################################################################################
    # Initialize the GNN.

    if(Load_From_File == True):
        # First, let's load the saved State
        State : Dict = torch.load("../saves/" + Load_File_Name);

        # Let's extract the stuff we need to initialize an object (widths, activations, pooling type).
        Conv_Widths         : List  = State["Conv Widths"];
        Conv_Type           : str   = State["Conv Type"];
        Conv_Activation     : str   = State["Conv Activation"];
        Pooling_Type        : str   = State["Pooling Type"];
        Pooling_Activation  : str   = State["Pooling Activation"];
        Linear_Widths       : List  = State["Linear Widths"];
        Linear_Activation   : str   = State["Linear Activation"];
        Output_Activation   : str   = State["Output Activation"];

        # Initialize the model.
        Model = GNN(Conv_Widths         = Conv_Widths, 
                    Linear_Widths       = Linear_Widths,
                    Conv_Type           = Conv_Type);
        
        # Now, load in the state dict.
        Model.Load_State(State);

    else:
        # Initialize the model using the settings above.
        Model = GNN(Net_Name            = f'{len(Conv_Widths)-1}_{Conv_Type}_Layers_{Pooling_Type}_Pool',
                    Conv_Widths         = Conv_Widths, 
                    Conv_Activation     = Conv_Activation,
                    Conv_Type           = Conv_Type,
                    Pooling_Type        = Pooling_Type,
                    Pooling_Activation  = Pooling_Activation,
                    Linear_Widths       = Linear_Widths,
                    Linear_Activation   = Linear_Activation,
                    Output_Activation   = Output_Activation);

    # Determine which device we're running on.
    Device : torch.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    Device = 'cpu'

    # Move the Model to that device.
    Model = Model.to(Device);


    ###################################################################################################
    # Set up Data.

    # Get the training, testing, and validation data sets.
    Train_Data  = LigandGraphIMDataset(root = '../data/train_dataset',  hdf5_file_name='postera_protease2_pos_neg_train.hdf5')
    Test_Data   = LigandGraphIMDataset(root = '../data/test_dataset',   hdf5_file_name='postera_protease2_pos_neg_test.hdf5')
    Val_Data    = LigandGraphIMDataset(root = '../data/val_dataset',    hdf5_file_name='postera_protease2_pos_neg_val.hdf5')

    # Move the datasets to the appropiate device.
    Train_Data.data.x           = Train_Data.data.x.to(Device);
    Train_Data.data.edge_index  = Train_Data.data.edge_index.to(Device);
    Train_Data.data.y           = Train_Data.data.y.to(Device);

    Test_Data.data.x            = Test_Data.data.x.to(Device);
    Test_Data.data.edge_index   = Test_Data.data.edge_index.to(Device);
    Test_Data.data.y            = Test_Data.data.y.to(Device);

    Val_Data.data.x             = Val_Data.data.x.to(Device);
    Val_Data.data.edge_index    = Val_Data.data.edge_index.to(Device);
    Val_Data.data.y             = Val_Data.data.y.to(Device);

    # Put the data into a data loader.
    Train_Loader    = torch_geometric.loader.DataLoader(   
                            Train_Data,
                            batch_size      = Batch_Size, 
                            shuffle         = True);

    Test_Loader     = torch_geometric.loader.DataLoader(   
                            Test_Data, 
                            batch_size      = Batch_Size, 
                            shuffle         = True);

    Val_Loader      = torch_geometric.loader.DataLoader(   
                            Val_Data, 
                            batch_size      = Batch_Size, 
                            shuffle         = True);


    ###################################################################################################
    # Select an optimizer.

    Optimizer   = torch.optim.Adam(Model.parameters(), lr = Learning_Rate);


    ###################################################################################################
    # Loop through the epochs!

    # Begin a timer.
    Epoch_Timer : float = time.perf_counter();
    print("Running %u epochs..." % Num_Epochs);

    # Initialize something to track the lowest training loss so far.
    Max_Correct    : int   = 0;
    Best_Model      : GNN   = Model.Copy();  

    for i in range(Num_Epochs):
        # Report epoch numnber
        print("Epoch %4u / %u" % (i + 1, Num_Epochs));

        # First, train!
        Train(  Model       = Model,
                Optimizer   = Optimizer,
                Loader      = Train_Loader,
                Lambda      = Lambda);

        # Now, test!
        Train_Results = Test(   Model   = Model, 
                                Loader  = Train_Loader,
                                Lambda  = Lambda);
        
        Val_Results = Test(     Model   = Model, 
                                Loader  = Test_Loader,
                                Lambda  = Lambda);

        # See if Model's validation loss is a new best. If so, track it.
        Num_Correct : int = Val_Results["True Positives"] + Val_Results["True Negatives"];
        if(Num_Correct > Max_Correct):
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
    print("an average of %7.2fs per epoch." % (Epoch_Runtime / Num_Epochs));  


    ###############################################################################################
    # Report testing, validation, and training loss for best model.

    # Evaluate the best model.
    Train_Results = Test(   Model   = Best_Model, 
                            Loader  = Train_Loader,
                            Lambda  = Lambda);
        
    Val_Results = Test(     Model   = Best_Model, 
                            Loader  = Val_Loader,
                            Lambda  = Lambda);

    Test_Results = Test(    Model   = Best_Model, 
                            Loader  = Test_Loader,
                            Lambda  = Lambda);

    print("\nBest model results:");
    print("Training:");
    Report_Test_Results(Train_Results);
    print("Validation:");
    Report_Test_Results(Val_Results);
    print("Testing:");
    Report_Test_Results(Test_Results);
 

    ##############################################################################################
    # Save the best model.

    # First, get the name of the file we're going to save to.
    Time                    = datetime.now();
    Save_File_Name  : str   = "../saves/" + Conv_Type + "_" + str(Time.day) + "_" + str(Time.hour) + "_" + str(Time.minute);
    Save_Model_Name  : str   = "../saves/" + Best_Model.Net_Name;
    
    # Now, get the Model state.
    State : Dict = Best_Model.Get_State();

    # Save it! 
    torch.save(State, Save_File_Name);
    torch.save(Best_Model, Save_Model_Name);


if __name__ == "__main__":
    main();
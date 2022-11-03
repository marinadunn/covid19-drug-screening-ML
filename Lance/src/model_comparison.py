
import  os
import  sys
import csv

from sklearn.metrics import accuracy_score

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




def Write_Results(File_Name : str, Model_Str : str, Accuracy : List[float], L2_Loss : List[float], Total_Loss : List[float]) -> None:
    # print(Model_Str)
    # print(os.getcwd())
    if not os.path.exists(File_Name):
        with open(File_Name, mode='w') as Csv_File:
            Csv_File.write('Model,Metric' + ','*len(Accuracy) + '\n')
        Csv_File.close()

    with open(File_Name, mode='a') as Csv_File: 
        Csv_File.write(Model_Str + ',Accuracy')
        for a in Accuracy:
            Csv_File.write(',' + str(a))
        Csv_File.write('\n')

        Csv_File.write(Model_Str + ',L2_Loss')
        for l in L2_Loss:
            Csv_File.write(',' + str(l))
        Csv_File.write('\n')

        Csv_File.write(Model_Str + ',Total_Loss')
        for t in Total_Loss:
            Csv_File.write(',' + str(t))
        Csv_File.write('\n')


def Get_Model_Str(Model_State : Dict) -> str:
    Model_Str = f'{len(Model_State["Conv Widths"]) - 1}_{Model_State["Conv Type"]}_Layers';
    Model_Str += f'_{Model_State["Pooling Type"]}_pool_lin';
    for lw in Model_State['Linear Widths']:
        Model_Str += ('_' + str(lw))

    return Model_Str


def main() -> None:     

    # GNN Model Settings Search Space
    Conv_Types : List[str]              = ['GCN', 'GraphSAGE', 'GAT'];

    Conv_Widths_List : List[List[int]] =    [
                                            [19, 10, 10, 10, 10, 10],
                                            [19, 10, 10, 10, 10],
                                            [19, 10, 10, 10],
                                            ];

    Conv_Activation : str              = 'elu';
    Pooling_Types : List[str]           = ['mean','max','add'];
    Pooling_Activation : str            = 'elu';

    Linear_Widths_List : List[List[int]]     =   [
                                            [32, 1],
                                            [16, 1]
                                            ];

    Linear_Activation : str            = 'elu';
    Output_Activation  : str            = 'sigmoid';


    # Training Settings
    Epochs      : int = 50;
    Lambda      : Dict  = {"Data" : 1.0, "L2" : 0.001};
    Batch_Size  : int = 64

    # Optimizer Settings
    Learning_Rate   : float = 0.01;
    Weight_Decay    : float = 5e-4;

    # Determine which device we're running on.
    Device : torch.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    Device = 'cpu'

    # Move the Model to that device.

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


    # Try each combination of the hyperparameter space and story results in csv
    for Conv_Type in Conv_Types:

        for Conv_Widths in Conv_Widths_List:

            for Pooling_Type in Pooling_Types:

                for Linear_Widths in Linear_Widths_List:
                    Tmp_Lin_List : List[int] = Linear_Widths[:]
                    Tmp_Lin_List.insert(0, sum(Conv_Widths[1:]))

                    Model : GNN = GNN(  Net_Name            = f'{len(Conv_Widths)-1}_{Conv_Type}_Layers_{Pooling_Type}_Pool',
                                        Conv_Widths         = Conv_Widths, 
                                        Conv_Activation     = Conv_Activation,
                                        Conv_Type           = Conv_Type,
                                        Pooling_Type        = Pooling_Type,
                                        Pooling_Activation  = Pooling_Activation,
                                        Linear_Widths       = Tmp_Lin_List,
                                        Linear_Activation   = Linear_Activation,
                                        Output_Activation   = Output_Activation);

                    Model = Model.to(Device);
                    Optimizer : torch.optim.Adam = torch.optim.Adam(Model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay);

                    print(f'Training {Model.Net_Name} GNN for {Epochs} Epochs');

                    Training_Accuracy       : List[float] = []
                    L2_Loss_Train           : List[float] = []
                    Total_Loss_Train        : List[float] = []
                    
                    Validation_Accuracy     : List[float] = []
                    L2_Loss_Valid           : List[float] = []
                    Total_Loss_Valid        : List[float] = []

                    Test_Accuracy           : List[float] = []
                    L2_Loss_Test            : List[float] = []
                    Total_Loss_Test         : List[float] = []

                    for i in range(Epochs):
                        print("Epoch %4u / %u" % (i + 1, Epochs));
                        


                        Train(  Model = Model,
                                Optimizer=Optimizer,
                                Loader = Train_Loader,
                                Lambda = Lambda);

                        Train_Results = Test(   Model   = Model, 
                                                Loader  = Train_Loader,
                                                Lambda  = Lambda);

                        Correct             : int = Train_Results["True Positives"] + Train_Results["True Negatives"];
                        Actual_Positives    : int = Train_Results["True Positives"] + Train_Results["False Negatives"];
                        Actual_Negatives    : int = Train_Results["True Negatives"] + Train_Results["False Positives"];

                        Training_Accuracy.append((Correct / (Actual_Negatives + Actual_Positives)).item())
                        L2_Loss_Train.append(Train_Results['L2 Loss'])
                        Total_Loss_Train.append(Train_Results['Total Loss'].item())

                        print("Training:");
                        Report_Test_Results(Train_Results);

                        if (i + 1) % 10 == 0: 
                            print("Validation:");
                            Val_Results = Test( Model   = Model, 
                                                Loader  = Val_Loader,
                                                Lambda  = Lambda);
                            Report_Test_Results(Val_Results);

                            Correct             : int = Val_Results["True Positives"] + Val_Results["True Negatives"];
                            Actual_Positives    : int = Val_Results["True Positives"] + Val_Results["False Negatives"];
                            Actual_Negatives    : int = Val_Results["True Negatives"] + Val_Results["False Positives"];

                            Validation_Accuracy.append((Correct / (Actual_Negatives + Actual_Positives)).item())
                            L2_Loss_Valid.append(Val_Results['L2 Loss'])
                            Total_Loss_Valid.append(Val_Results['Total Loss'].item())

                            print('Test:');
                            Test_Results = Test( Model   = Model, 
                                                Loader  = Test_Loader,
                                                Lambda  = Lambda);
                            Report_Test_Results(Test_Results);

                            Correct             : int = Test_Results["True Positives"] + Test_Results["True Negatives"];
                            Actual_Positives    : int = Test_Results["True Positives"] + Test_Results["False Negatives"];
                            Actual_Negatives    : int = Test_Results["True Negatives"] + Test_Results["False Positives"];
                            Test_Accuracy.append((Correct / (Actual_Negatives + Actual_Positives)).item());
                            L2_Loss_Test.append(Test_Results['L2 Loss']);
                            Total_Loss_Test.append(Test_Results['Total Loss'].item());


                    Model_Str = Get_Model_Str(Model.Get_State());
                    Write_Results('Train_Results.csv', Model_Str, Training_Accuracy, L2_Loss_Train, Total_Loss_Train);
                    Write_Results('Validation_Results.csv', Model_Str, Validation_Accuracy, L2_Loss_Valid, Total_Loss_Valid);
                    Write_Results('Test_Results.csv', Model_Str, Test_Accuracy, L2_Loss_Test, Total_Loss_Test);
                            
                    
if __name__ == '__main__':
    main()
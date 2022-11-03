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

model1 = torch.load('../saves/5_GCN_Layers_max_Pool')
model2 = torch.load('../saves/4_GCN_Layers_mean_Pool')
model3 = torch.load('../saves/5_GAT_Layers_mean_Pool')
model4 = torch.load('../saves/5_GraphSAGE_Layers_mean_Pool')
model5 = torch.load('../saves/4_GraphSAGE_Layers_mean_Pool')

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

Device : torch.device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu');

Test_Data   = LigandGraphIMDataset(root = '../data/test_dataset',   hdf5_file_name='postera_protease2_pos_neg_test.hdf5')

Test_Data.data.x            = Test_Data.data.x.to(Device);
Test_Data.data.edge_index   = Test_Data.data.edge_index.to(Device);
Test_Data.data.y            = Test_Data.data.y.to(Device);

Test_Loader     = torch_geometric.loader.DataLoader(   
                        Test_Data, 
                        batch_size      = 8, 
                        shuffle         = True);


Num_Data        : int   = 0;

True_Positives  : int   = 0;
False_Negatives : int   = 0;
True_Negatives  : int   = 0;
False_Positives : int   = 0;

Total_Data_Loss : float = 0;

# Loop through the batches.
with torch.no_grad():
    for Data in Test_Loader:
        # Pass the data through the Model to get predictions.
        # Pred1 = torch.round(model1(Data).reshape(-1)).to(torch.int32);
        # Pred2 = torch.round(model2(Data).reshape(-1)).to(torch.int32);
        # Pred3 = torch.round(model3(Data).reshape(-1)).to(torch.int32);
        # Pred4 = torch.round(model4(Data).reshape(-1)).to(torch.int32);
        # Pred5 = torch.round(model5(Data).reshape(-1)).to(torch.int32);
        Pred1 = model1(Data).reshape(-1);
        Pred2 = model2(Data).reshape(-1);
        Pred3 = model3(Data).reshape(-1);
        Pred4 = model4(Data).reshape(-1);
        Pred5 = model5(Data).reshape(-1);

        ensemble_pred = torch.add(Pred1, Pred2)
        ensemble_pred = torch.add(ensemble_pred, Pred3)
        ensemble_pred = torch.add(ensemble_pred, Pred4)
        ensemble_pred = torch.add(ensemble_pred, Pred5)
        ensemble_pred = torch.div(ensemble_pred,5)
        # print(ensemble_pred)

        # Update number of data points.
        Num_Data += torch.numel(ensemble_pred);

        # Round the predictions.
        Rounded_Predictions : torch.Tensor = torch.round(ensemble_pred).to(torch.int32);

        # Cast y to int32.
        y : torch.Tensor = Data.y.to(torch.int32);

        # Determine which predictions are correct!
        Correct_Predictions : torch.Tensor = torch.eq(Rounded_Predictions, y);
        # print(Correct_Predictions)
        # print(y)

        # Count the number of true/false positives/negatives
        True_Positives  += torch.sum(torch.logical_and(Correct_Predictions,                     y)); 
        False_Positives += torch.sum(torch.logical_and(torch.logical_not(Correct_Predictions),  torch.logical_not(y)));
        True_Negatives  += torch.sum(torch.logical_and(Correct_Predictions,                     torch.logical_not(y)));
        False_Negatives += torch.sum(torch.logical_and(torch.logical_not(Correct_Predictions),  y));

correct = True_Positives + True_Negatives
actual_positive = True_Positives + False_Negatives
actual_negative = True_Negatives + False_Positives

accuracy = correct / (actual_positive + actual_negative)

print('Ensemble Model Accuracy:', accuracy.item())
# print('True Positives', True_Positives)
# print('False Positives', False_Positives)
# print('True Negatives', True_Negatives)
# print('False Negatives', False_Negatives)


import seaborn as sns
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(4, 4))

cf_matrix = [[int(True_Positives.item()), False_Negatives.item()],
             [False_Positives.item(), True_Negatives.item()]]

ax = sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

ax.set_title('Test Data');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

## Display the visualization of the Confusion Matrix.
plt.tight_layout()
plt.savefig('ensemble_model_test_conf_mat.png')
# print('Actual Positive:', actual_positive.item())
# print('Actual Negative:', actual_negative.item())
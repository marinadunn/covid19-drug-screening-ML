from re import S
import  h5py        as      h5;
import  numpy;
import  os;
import  torch;
from    typing      import  Dict, List;

from torch_geometric.loader     import DataLoader;
from torch_geometric.data       import Data, InMemoryDataset;
from torch_geometric.utils      import dense_to_sparse;
from sklearn.metrics            import pairwise_distances;



# Settings.
Bond_Threshold          : float = 1.8;
Max_Bars                : int   = 50;
Track_Stats             : bool  = True; 
Keep_Spatial_Features   : bool  = False;



class LigandGraphIMDataset(InMemoryDataset):
    def __init__(   self, 
                    root            : str, 
                    hdf5_file_name  : str, 
                    transform       = None, 
                    pre_transform   = None, 
                    pre_filter      = None) -> None:
        """
        This function intiailizes a data loader using the name of the hdf5 file which contains the
        data, as well as the directory which houses that file. 

        ------------------------------------------------------------------------------------------
        Arguments:

        root: A string that lists the path (relative to the data directory) to the directory that
        houses the hdf5 file we want to extract.

        hdf5_file_name: A string containing the name of the hdf5 file we want to extract data from.
        """
            
        # Record the hdf5 file name.
        self.hdf5_file_name : str = hdf5_file_name;

        # Call the InMemoryDataset constructor. We need to pass the root, transformers, and filters
        super().__init__(root, transform, pre_transform, pre_filter);

        # Load the processed data. 
        self.data, self.slices = torch.load(self.processed_paths[0])



    @property
    def raw_file_names(self) -> List[str]:
        """ This function must return a list of files that must be present in the raw directory 
        for the dataloader to work. 
        
        -------------------------------------------------------------------------------------------
        Returns: 

        A list whose lone element holds the name of the hd5 file that contains the raw, 
        unprocessed data.
        """

        if self.hdf5_file_name != None:
            return [self.hdf5_file_name];
        else:
            return [];



    @property
    def processed_file_names(self) -> List[str]:
        """
        This function returns a list whose lone element holds the name of the file that houses the 
        processed data. torch_geometric uses this to determine if it needs to process the data. In 
        particular, if the returned string corresponds to an existing file, the data loader does 
        not process the data

        -------------------------------------------------------------------------------------------
        Returns: 

        A list whose lone element holds the name of the file in which we store the processed data
        """

        return ['data.pt'];



    def download(self):
        """
        This function is supposed to download the data if it is not present in the raw directory.
        We have no way of doing that, so this function just raises an error if called.
        """

        raise NotImplementedError("download is not defined for this function. You must have passed the wrong raw file name.")



    def process(self) -> None:
        """
        This function, which has no arguments and returns nothing, processes the raw data into 
        processed data.  
        """

        # This is a list whose ith entry holds the Data object associated with the ith ligand.
        Data_List : List = [];

        # Open the hdf5 file. 
        hdf5_file_path  = os.path.join(self.raw_dir, self.hdf5_file_name);
        hdf5_file       = h5.File(hdf5_file_path, 'r');

        # The hdf5 file is structured as a dictionary, with a key for each molecule. We want to
        # load the molecules one-at-a-time.
        Molecule_Names  : List[str] = list(hdf5_file.keys());
        Num_Molecules   : int       = len(Molecule_Names);



        # Loop through the molecules.
        for m in range(Num_Molecules):
            # Report how many molecules we've processed.
            Num_Bars        : int = int((m/Num_Molecules)*Max_Bars);
            Progress_Bar    : str = "[" + "#"*Num_Bars + " "*(Max_Bars - Num_Bars) + "]"
            print(Progress_Bar, end = '');
            print(" (%5u / %u)" % (m, Num_Molecules), end = '\r');

            # Extract the name of the ith molecule.
            Molecule_Name : str = Molecule_Names[m];

            # First, extract the ligand data. This is a 100 x 22 numpy ndarray.   
            Ligand_Data : numpy.ndarray = hdf5_file[Molecule_Name]['ligand'];

            # Get ligand groud_truth label (0 = no bind, 1 = bind)
            Label : int = hdf5_file[Molecule_Name].attrs['label'];

            # Each ligand is stored in a file with 100 rows. Each row corresponds to an atom. Since
            # most of the ligands have < 100 atoms, most of the rows are unused (do not correspond
            # to an atom). They fill unused rows with 0s. First, need to find the first row of 
            # zeros
            Num_Atoms : int = 0;
            for i in range(Ligand_Data.shape[0]):
                # Check if the current row contains all zeros (equivalently, the sum of the abs
                # value of its entries is zero)
                if(numpy.sum(numpy.abs(Ligand_Data[i, :])) == 0):
                    # If so, we now know the number of atoms.  
                    Num_Atoms = i;
                    break;
            
            # Now, only keep the bits of the ligand data corresponding to actual atoms.
            Ligand_Data = Ligand_Data[0:Num_Atoms, :];

            # Set up buffers to track statistics.
            if(Track_Stats and m == 0):
                if(Keep_Spatial_Features == True):
                    Num_Features    : int = Ligand_Data.shape[1];
                else:
                    Num_Features    : int = Ligand_Data.shape[1] - 3;


                Total_Atoms     : int = 0;
                Running_Sum     : torch.Tensor = torch.zeros(Num_Features, dtype = torch.float32);
                Running_Max     : torch.Tensor = torch.zeros(Num_Features, dtype = torch.float32);
                Running_Min     : torch.Tensor = torch.zeros(Num_Features, dtype = torch.float32);

            # Extract the coordinates of the atoms in the molecule.
            Coords      = Ligand_Data[:, 0:3];
        
            # This returns a Num_Atoms x Num_Atoms matrix whose i, j entry holds the euclediean 
            # distance from atom i to atom j. Specifically, if atom i and j have coordinates 
            # x, y \in \mathbb{R}^3, respectively, then the i,j entry of dists is ||x - y||_2. 
            # This means that dists is a symmetric matrix.
            Dists : numpy.ndarray = pairwise_distances(Coords, metric = 'euclidean');

            # We want to determine which pairs of atoms have a bond between them. In our case, 
            # any pair of atoms with a distance of < Bond_Threshold have a bond. Thus, we can 
            # use an element-wise check of the elements of dists to make the adjacency matrix.
            A : numpy.ndarray = numpy.less(Dists, Bond_Threshold);

            # Next, we need to convert the adjacency matrix into a 2 x |E| array, where |E| is the
            # number of edges. 
            Num_Edges : int = numpy.sum(A);
            
            # Next, we loop through all possible pairs of edges. For each pair, we add an edge
            # if the corresponding element of the adjacency matrix is 1.
            Edge_Index      : torch.Tensor = torch.empty(size = (2, Num_Edges), dtype = torch.long);
            Edge_Counter    : int          = 0;

            for i in range(Num_Atoms):
                for j in range(Num_Atoms):
                    # Check if there is an edge from node j to node i. If so, set the next column
                    # of edge_index to [j, i].
                    if(A[i, j] == 1):
                        Edge_Index[0, Edge_Counter] = j;
                        Edge_Index[1, Edge_Counter] = i;
                        Edge_Counter += 1; 

            # Now, set up the feature vector matrix. This can either hold the coordinates or not.
            # Recall that at this stage, ligand_data is a Num_Atoms x 22 matrix and the first 3
            # features are the atom coordinates. Thus, if we do not spatial information, we only 
            # use features 3-21.
            if(Keep_Spatial_Features == True):
                Feature_Matrix : torch.Tensor   = torch.tensor(Ligand_Data);
            else: # Keep_Spatial_Features == False
                Feature_Matrix : torch.Tensor   = torch.tensor(Ligand_Data[:, 3:]);

            # Update statistics.
            if(Track_Stats):
                Total_Atoms += Num_Atoms;
                Running_Sum += torch.sum(Feature_Matrix, dim = 0);

                for j in range(Num_Features):
                    Running_Max[j] = torch.max(torch.max(Feature_Matrix[:, j]), Running_Max[j]);
                    Running_Min[j] = torch.min(torch.min(Feature_Matrix[:, j]), Running_Min[j]);

            # Now, initialize a Data object using the extracted  
            Graph_Data = Data(  x           = Feature_Matrix, 
                                edge_index  = Edge_Index,
                                y           = Label);

            # Append this graph's Data object to the Data_List.
            Data_List.append(Graph_Data);

        # Report final statistics.
        if(Track_Stats):
            # Compute the per-feature means.
            Means : torch.Tensor = numpy.divide(Running_Sum, Total_Atoms);

            # Report per-feature min.
            print("Min | ", end = '');
            for i in range(Num_Features):
                print("%6.3f" % Running_Min[i].item(), end = '');
                if(i != Num_Features - 1):
                    print(",", end = '');
            print();

            # Report per-feature mean.
            print("Mean| ", end = '');
            for i in range(Num_Features):
                print("%6.3f" % Means[i].item(), end = '');
                if(i != Num_Features - 1):
                    print(",", end = '');
            print();

            # Report per-feature max.
            print("Max | ", end = '');
            for i in range(Num_Features):
                print("%6.3f" % Running_Max[i].item(), end = '');
                if(i != Num_Features - 1):
                    print(",", end = '');
            print();

        # Report that we are done!
        Progress_Bar : str = "[" + "#"*Max_Bars + "]";        
        print(Progress_Bar, end = '');
        print("(%5u / %5u)" % (Num_Molecules, Num_Molecules));

        # We are done with the hdf5 file, so close it.
        hdf5_file.close();

        # Prepare the Data_List for serialization... then serialize it.
        data, slices = self.collate(Data_List);

        # Replace y's data attribute (which is a list) with an equivalent 1D tensor.
        data.y = torch.tensor(data.y, dtype = torch.float32);

        torch.save((data, slices), self.processed_paths[0]);





def main():
    # Generate the three data sets.
    train_dataset = LigandGraphIMDataset(
                        root            = './train_dataset', 
                        hdf5_file_name  = 'postera_protease2_pos_neg_train.hdf5');

    test_dataset = LigandGraphIMDataset(
                        root            = './test_dataset', 
                        hdf5_file_name  = 'postera_protease2_pos_neg_test.hdf5');

    val_dataset = LigandGraphIMDataset(
                        root            = './val_dataset', 
                        hdf5_file_name  = 'postera_protease2_pos_neg_val.hdf5');
  
    """
    loader = DataLoader(train_dataset, batch_size = 32);
    total = 0
    for data in loader:
        total += 32;

    print(total)
    """



if __name__ == '__main__':
    main()

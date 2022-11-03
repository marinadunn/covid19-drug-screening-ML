from audioop import bias
from multiprocessing.sharedctypes import Value
import  torch;
import  torch_geometric;
from    typing                      import  List, Dict, Callable;



class GNN(torch.nn.Module):
    """
    TO DO 

    class doc string.
    """

    def __init__(   self, 
                    Conv_Widths         : List[int],
                    Linear_Widths       : List[int],
                    Conv_Type           : str,
                    Conv_Activation     : str       = "elu",
                    Pooling_Type        : str       = "max",
                    Pooling_Activation  : str       = "elu",
                    Linear_Activation   : str       = "elu",
                    Output_Activation   : str       = "sigmoid"):
        """
        This sets up a Graph Neural Network. The GNN has a set of convolutional layers, one pooling 
        layer, and a set of linear layers. Each kind of layer (as well as the final output) can 
        have an activation function as well. There are a lot of arguments, but there need to be 
        because this is a complicated architecture.

        -------------------------------------------------------------------------------------------
        Arguments:

        Conv_Widths: This is a list of integers. The ith entry of this list specifies the number of 
        features in the node feature vectors after the ith convolutional layer. If the ith 
        entry is n and the i + 1th is m, then the ith conv layer takes in node feature vectors with
        n features and produces ones with m. 

        Linear_Widths: A list of integers whose ith entry specifies the width of the ith linear 
        layer (with the 0th being the output of the Pooling layer).

        Conv_Type: A string that specifies the type of convolutional layers we want to use. Options
        are "GCN", "GraphSAGE", and "GAT".

        Conv_Activation: This is a string that specifies the activation function after each 
        convolutional layer.

        Pooling_Type: This is a string that specifies the type of pooling layer we want to use
        after the Conv layers.

        Pooling_Activation: This is a string that specifies the activation function for the pooling
        layer.

        Linear_Activations: A string that specifies the activation function for the linear layers 

        Output_Activation: A string that specifies the activation function we use on the output of 
        the last layer.
        """ 

        # Run checks.
        for i in range(len(Conv_Widths)):
            assert(Conv_Widths[i] > 0);
        for i in range(len(Linear_Widths)):
            assert(Linear_Widths[i] > 0);

        # Call the Module initializer.
        super(GNN, self).__init__();

        # Set up the convolutional layers.
        self.Conv_Widths        : List[int] = Conv_Widths;
        self.Num_Conv_Layers    : int       = len(Conv_Widths) - 1;
        self.Conv_Layers        : List      = torch.nn.ModuleList();
        self.Conv_Type          : str       = Conv_Type.strip();

        for i in range(self.Num_Conv_Layers):
            if(self.Conv_Type.lower() == "gcn"):
                self.Conv_Layers.append(torch_geometric.nn.GCNConv( 
                            in_channels     = Conv_Widths[i],
                            out_channels    = Conv_Widths[i + 1],
                            bias            = False));
            elif(self.Conv_Type.lower() == "graphsage"):
                self.Conv_Layers.append(torch_geometric.nn.SAGEConv( 
                            in_channels     = Conv_Widths[i],
                            out_channels    = Conv_Widths[i + 1],
                            bias            = False));
            elif(self.Conv_Type.lower() == "gat"):
                self.Conv_Layers.append(torch_geometric.nn.GATConv( 
                            in_channels     = Conv_Widths[i],
                            out_channels    = Conv_Widths[i + 1],
                            bias            = False));
            else:
                raise ValueError("Invalid conv type. Got %s" % Conv_Type);

        # Set the activation function for the convolutional layers.
        self.Conv_Activation = self._Get_Activation_Function(Encoding = Conv_Activation);

        # Add the pooling layer.
        self.Pooling_Layer = self._Get_Pooling_Function(Pooling_Type);

        # Set the activation function for the pooling layer.
        self.Pooling_Activation = self._Get_Activation_Function(Encoding = Pooling_Activation);

        # Add the linear layers.
        self.Linear_Widths      : List[int] = Linear_Widths;
        self.Num_Linear_Layers  : int       = len(Linear_Widths) - 1;
        self.Linear_Layers      : List      = torch.nn.ModuleList(); 

        for i in range(self.Num_Linear_Layers):
            self.Linear_Layers.append(torch.nn.Linear(
                                            in_features     = Linear_Widths[i],
                                            out_features    = Linear_Widths[i + 1]));
        
        # Initialize the weight matrices, bias vectors in the linear layers.
        for i in range(self.Num_Linear_Layers):
            torch.nn.init.xavier_uniform_(self.Linear_Layers[i].weight);
            torch.nn.init.zeros_(self.Linear_Layers[i].bias);

        # Set the activation function for the linear layers.
        self.Linear_Activation = self._Get_Activation_Function(Encoding = Linear_Activation);

        # Finally, set the activation function for the output layer.
        self.Output_Activation = self._Get_Activation_Function(Encoding = Output_Activation);



    def _Get_Pooling_Function(self, Encoding : str) -> Callable:
        """
        This function returns the activation function corresonding to the Encoding.

        -------------------------------------------------------------------------------------------
        Argumnets:

        Encoding: A string. This should be the name of an activation function in torch, or the 
        word "none".

        -------------------------------------------------------------------------------------------
        Returns:

        The pooling layer associated with the Encoding.
        """

        # First, strip the encoding and make it lower case (this makes it easier to parse)
        Encoding = Encoding.strip().lower();

        if  (Encoding == "add"):
            return torch_geometric.nn.global_add_pool;
        elif(Encoding == "mean"):
            return torch_geometric.nn.global_mean_pool;
        elif(Encoding == "max"):
            return torch_geometric.nn.global_max_pool;
        else:
            raise ValueError("Invalid Pooling string. Valid options are \"add\", \"max\", and \"mean\". Got %s" % Encoding);



    def _Get_Pooling_String(self, f : Callable) -> str:
        """
        This function returns the string associated with the  pooling layer function, f. The 
        returned string can be fed into _Get_Pooling_Function to recover the original 
        activation  function. Thus, this function is the inverse of _Get_Pooling_Function.

        -------------------------------------------------------------------------------------------
        Argumnets:

        Encoding: A string. This should be the name of an pooling layer in torch_geometric.nn.

        -------------------------------------------------------------------------------------------
        Returns:

        The string associated with the pooling function f.
        """

        # Return the corresponding string.
        if  (f == torch_geometric.nn.global_add_pool):
            return "add";
        elif(f ==  torch_geometric.nn.global_mean_pool):
            return "mean";
        elif(f ==  torch_geometric.nn.global_max_pool):
            return "max";
        else:
            raise ValueError("Invalid Pooling function."); 



    def _Get_Activation_Function(self, Encoding : str) -> Callable:
        """
        This function returns the activation function corresonding to the Encoding.

        -------------------------------------------------------------------------------------------
        Argumnets:

        Encoding: A string. This should be the name of an activation function in torch, or the 
        word "none".

        -------------------------------------------------------------------------------------------
        Returns:

        The activation function with the Encoding.
        """

        # First, strip the encoding and make it lower case (this makes it easier to parse)
        Encoding = Encoding.strip().lower();

        # Now, parse it.
        if  (Encoding == "none"):
            return None;
        elif(Encoding == "relu"):
            return torch.relu;
        elif(Encoding == "sigmoid"):
            return torch.sigmoid;
        elif(Encoding == "tanh"):
            return torch.tanh;
        elif(Encoding == "elu"):
            return torch.nn.functional.elu;
        elif(Encoding == "softmax"):
            return torch.nn.functional.softmax;  
        else:
            raise ValueError("Invalid activation function string. Got %s" % Encoding);    



    def _Get_Activation_String(self, f : Callable):
        """
        This function returns the string associated with the activation function f. The returned
        string can be fed into _Get_Activation_Function to recover the original activation 
        function. Thus, this function is the inverse of _Get_Activation_Function.

        -------------------------------------------------------------------------------------------
        Argumnets:

        f: A function. This should be an torch activation function returned by 
        _Get_Activation_Function

        -------------------------------------------------------------------------------------------
        Returns:

        The string associated with the activation function f.
        """

        # Parse the activation function, return the corresponding string.
        if  (f is None):
            return "none";
        elif(f == torch.relu):
            return "relu";
        elif(f == torch.sigmoid):
            return "sigmoid";
        elif(f == "tanh"):
            return torch.tanh;
        elif(f == torch.nn.functional.elu):
            return "elu";
        elif(f == torch.nn.functional.softmax):
            return "softmax";
        else:
            raise ValueError("Unknown activation function.");



    def forward(self, data) -> torch.Tensor:
        """
        A GNN object represents a mapping on a graph. This is that mapping. It takes in a data 
        object (which should be batch returned by a Torch_Geometric dataloader) and evaluates 
        self on the graphs in that batch.

        -------------------------------------------------------------------------------------------
        Arguments:

        data: A batch returned by a dataloader object.

        -------------------------------------------------------------------------------------------
        Returns: 

        a tenosr whose ith entry holds the image under self of the ith graph in data.
        """

        # First, extract the feature matrix, edge index info, and batch info.
        x           : torch.Tensor  = data.x;
        edge_index  : torch.Tensor  = data.edge_index;
        batch       : torch.Tensor  = data.batch;

        # Define a list to hold the feature vectors at each step.
        r = [x];

        # Pass x through the convolutional layers.
        for i in range(self.Num_Conv_Layers):
            # Pass x through the ith convolutional layer.
            y   = self.Conv_Layers[i](r[i], edge_index);

            # Pass y through the activation function for the convolutional layers, if there is one.
            if(self.Conv_Activation is not None):
                r.append(self.Conv_Activation(y));
            else:
                r.append(y);
        
        Combined_r = torch.hstack(r[1:]);
        

        # Pass combined r through the pooling layer.
        x = self.Pooling_Layer(Combined_r, batch);

        if(self.Pooling_Activation is not None):
            x = self.Pooling_Activation(x);
        else:
            x = y;

        # Pass x through the linear layers.
        for i in range(self.Num_Linear_Layers - 1):
            y = self.Linear_Layers[i](x);

            if(self.Linear_Activation is None):
                x = y;
            else:
                x = self.Linear_Activation(y);
        
        # Finally, pass x through the final layer and apply the output activation function.
        y = self.Linear_Layers[-1](x);

        if(self.Output_Activation is None):
            return y;
        else:
            return self.Output_Activation(y);


    

    def Get_State(self) -> Dict:
        """
        This function returns a dictionary that houses everything necessary to fully define this
        object. In particular, the returned dictionary contains, in essence, a state dict of state 
        dicts, together with additional information. In particular, the dictionary contains an entry
        each convolutional/linear layer. The corresponding value is the state dict for that layer. 
        The dictionary also contains strings that characterize the pooling layer, the activation 
        functions, and the convolutional/linear widths. In essence, the returned dictionary is a 
        "super" state dictionary. The returned dictionary can be serialized using torch's save 
        method.

        -------------------------------------------------------------------------------------------
        Returns:

        The dictionary described above.
        """

        # Initialize the dictionary using the strings associated with the network's activation 
        # functions. Also throw in the Widths and the pooling layer.
        State   = { "Conv Activation"       : self._Get_Activation_String(self.Conv_Activation),
                    "Conv Type"             : self.Conv_Type,
                    "Pooling Activation"    : self._Get_Activation_String(self.Pooling_Activation),
                    "Linear Activation"     : self._Get_Activation_String(self.Linear_Activation),
                    "Output Activation"     : self._Get_Activation_String(self.Output_Activation),
                    "Conv Widths"           : self.Conv_Widths,
                    "Linear Widths"         : self.Linear_Widths,
                    "Pooling Type"          : self._Get_Pooling_String(self.Pooling_Layer)};
        
        # Now, append the state dicts for the conv layers.
        for i in range(self.Num_Conv_Layers):
            # Get the key for this item. 
            Key : str   = "Conv " + str(i);

            # Append its state dictionary to the State dict.
            State[Key]  = self.Conv_Layers[i].state_dict();
        
        # Finally, append the state dicts for the linear layers.
        for i in range(self.Num_Linear_Layers):
            # Get the key for this item. 
            Key : str   = "Linear " + str(i);

            # Append its state dictionary to the State dict.
            State[Key]  = self.Linear_Layers[i].state_dict();
        
        # All done!
        return State;



    def Load_State(self, State : Dict) -> None:
        """
        This function sets self's attributes to match those in the State dictionary. The State
        argument should be a dictionary returned by the Get_State method in this class. We update
        self's activation functions and tensors to match those saved in State. Note that the 
        number of conv/linear layers in self, as well as their widths, must match those in State.
        If not, this will throw an error.

        Note: This method can only work if self has the same linear/conv widths as the network 
        in State. Likewise, self must use the same conv type as the network in State.

        -------------------------------------------------------------------------------------------
        Arguments:

        State: A GNN State dictionary. This should be a dictionary returned by the Get_State
        function of the GNN class.
        """

        # Run checks.
        assert(len(self.Conv_Widths)    == len(State["Conv Widths"]));
        assert(len(self.Linear_Widths)  == len(State["Linear Widths"]));
        assert(self.Conv_Type           == State["Conv Type"]);

        for i in range(len(self.Conv_Widths)):
            assert(self.Conv_Widths[i]  == State["Conv Widths"][i]);

        for i in range(len(self.Linear_Widths)):
            assert(self.Linear_Widths[i] == State["Linear Widths"][i]);

        # First, let's update the activation functions.
        self.Conv_Activation    = self._Get_Activation_Function(State["Conv Activation"]); 
        self.Pooling_Activation = self._Get_Activation_Function(State["Pooling Activation"]); 
        self.Linear_Activation  = self._Get_Activation_Function(State["Linear Activation"]); 
        self.Output_Activation  = self._Get_Activation_Function(State["Output Activation"]); 

        # Next, let's set the pooling layer.
        self.Pooling_Layer      = self._Get_Pooling_Function(State["Pooling Type"]);

        # Now load each conv layer's state dict.
        for i in range(self.Num_Conv_Layers):
            # Get the layer key.
            Key : str = "Conv " + str(i);

            # Update the layer's state dict from State
            self.Conv_Layers[i].load_state_dict(State[Key]);

        # Finally, load each linear layer's state dict.
        for i in range(self.Num_Linear_Layers):
            # Get the layer key.
            Key : str = "Linear " + str(i);

            # Update the layer's state dict from State
            self.Linear_Layers[i].load_state_dict(State[Key]);
    


    def Copy(self):
        """
        This function returns a deep copy of self.
        
        -------------------------------------------------------------------------------------------
        Returns:

        A Network object that is, for all intents and purposes, an identical copy of self, but 
        stored in a different location. Specifically, the returned network has the same 
        architecture, weights, and biases of self, but its weights/biases are stored in a different
        location in memory. As such, changes to either networks weights/biases will not impact the 
        others.        
        """

        # First, let's get our state dictionary
        State : Dict = self.Get_State();

        # Now, let's make a new network object.
        Copy : GNN = GNN(   Conv_Widths         = State["Conv Widths"],
                            Conv_Type           = State["Conv Type"],
                            Linear_Widths       = State["Linear Widths"]);
        
        # load in the saved copy. 
        Copy.Load_State(State);

        # All done... return Copy!
        return Copy;
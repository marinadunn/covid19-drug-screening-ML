# Nonsense to add Code diectory to the python search path.
import os;
import sys;

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the src directory to the python path.
src_Path   = os.path.join(parent_dir, "src");
sys.path.append(src_Path);

import  unittest;
import  random;
import  numpy;

from    Subset  import Generate_Index_Subsets;



Verbose : bool = True;



class Test_Subset(unittest.TestCase):
    def test_Subset(self):
        ###########################################################################################
        # Setup.

        # First, let's pick a subset size/number of subsets.
        Superset_Size   : int = random.randint(100, 1000);
        Subset_Size     : int = random.randint(10, min(Superset_Size - 1, 200));
        Num_Subsets     : int = random.randint(50, 500);

        if(Verbose):
            print("Superset Size    = %u" % Superset_Size);
            print("Subset Size      = %u" % Subset_Size);
            print("Num Subsets      = %u" % Num_Subsets);

        # Now, let's make the subsets.
        Subsets : numpy.ndarray = Generate_Index_Subsets(   Superset_Size   = Superset_Size, 
                                                            Subset_Size     = Subset_Size, 
                                                            Num_Subsets     = Num_Subsets);
        

        ###########################################################################################
        # Test 0: Shape

        # Make sure it has the right size. 
        self.assertEqual(Subsets.shape[0], Num_Subsets);
        self.assertEqual(Subsets.shape[1], Subset_Size);


        ###########################################################################################
        # Test 1: Subsets

        # Now make sure that each row contains a valid subset of {1, 2, ... , Superset_Size}. 
        # To do this, we make sure each row of Subsets contains unique elements. How do we do
        # this? By keeping track of how many times each number in {1, 2, ... , Superset_Size} 
        # appears in Subsets. We begin by making an array called "Counts", which has shape 
        # 2 x Superset_Size and is intiailized to hold zeros. We then cycle through the rows
        # of Subsets. Let i denote the current row index. We assume that, at the start of the 
        # ith step, both rows of Counts are identical, with their jth entries holding the 
        # number of times that the number j appears in the first i - 1 rows of Subset. We then 
        # cycle through the entries of the ith row of Subsets. If the jth entry is equal to 
        # n, we assert that Counts[0, n] == Counts[1, n], and then increment Counts[1, n]. After 
        # cycling through the row, we set Counts[0, :] = Counts[1, :]. The only way we can pass
        # the ith step is if each entry in the ith row of Subsets is unique (so each index is 
        # incremeneted at most one time). After checking, this array also holds the number of 
        # times each element of {1, 2, ... , Superset_Size} appears in Subsets. 
        Counts : numpy.ndarray = numpy.zeros(shape = (2, Superset_Size), dtype = numpy.int32);

        for i in range(Num_Subsets):
            for j in range(Subset_Size):
                # Get i,j entry of Subsets.
                n : int = Subsets[i, j];
                
                # Make sure this number has appeared at most one time in the ith row.
                self.assertEqual(Counts[0, n], Counts[1, n]);

                # Increment Counts[1, n].
                Counts[1, n] += 1;
            
            # Now, update the 0 row of Counts to match the 1 row of Counts.
            Counts[0, :] = Counts[1, :];
        
        if(Verbose):
            print("Final Counts:");
            print(Counts[0, :]);
        

if __name__ == "__main__":
    unittest.main();
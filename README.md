# Auto-GINClus: Self-supervised clustering and identification of RNA structural motifs using GVP-GINE autoencoder
###### Authors: Created by Nabila Shahnaz Khan in collaboration with Cuncong Zhong and Shaojie Zhang
Auto-GINClus source code is implemented using __Python 3.11.4__, __PyTorch 2.6.1__, __PyTorch Geometric 2.6.1__ and can be executed in __64-bit Linux__ machine. For given RNA structural motifs (RSMs) and RNA motif PDB locations, Auto-GINClus generates the graph representation for each motif/loop. Then, it generates subclusters based on their structural (base interaction and 3D structure) similarity. It also generates Q-score for each subcluster and identies new RSMs based on thesholds Q and R. It can also generate side-by-side and superimposed images of motifs/loops for each subcluster.  



## Install Instructions 


### Installation using Docker
To simplify installation process, a [Dockerfile](Dockerfile) has been provided. This will automatically install virtual environment for installing python dependencies, prevent conflicts between dependencies and keep application environement clean.   

#### Install Docker:
For Windows and macOS, install Docker Desktop by downloading it from Docker's official website. Install Docker on your Linux OS using the following commands:  
```
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker 
```

#### Run using Dockerfile
After downloading/cloning Auto-GINClus from Github, run the following commands inside the downloaded Auto-GINClus folder to build the docker file image and then to run Auto-GINClus.  

\# Build an image
```
docker build -t autoginclus_container .
```

\# Run inside a container
```
docker run -it --rm autoginclus_container bash
```

### Installation without Docker
#### Install python3:
```
Debian/Ubuntu: apt install python3.11  
Fedora/CentOS: dnf install python3.11 
```

#### Install pip3: 
```
Debian/Ubuntu: apt install python3-pip  
Fedora/CentOS: dnf install python3-pip  
```

#### Install required Python libraries:  
The python libraries required to run Auto-GINClus are included in the [requirements.txt](requirements.txt) file. To install all required python libraries, please navigate to the Auto-GINClus home directory in the terminal and execute the following command.
```
pip3 install -r requirements.txt
```

#### Existing python packages:  
os, sys, shutil, math, random, subprocess, glob, time, argparse, logging, requests  
  
*** If any of the above mentioned package doesn't exist, then please install with command 'pip3 install package-name' ***  


#### Create Executables:
After downloading .zip folder from GitHub, the executable permission of the binary files might get revoked. Run the following command inside Auto-GINClus folder to create executables.  
```
chmod + src/my_lib/TMalign/TMalign-20180426/TMalign
```

#### Create virtual environment (optional): 
For latest version of Ubuntu (>=23.0), all python packages have to be installed under a virtual environment. Use the follwing instructions to create a virtual envionment folder to run requirements.txt and Auto-GINClus code inside it:

```
mkdir Auto-GINClus_venv  
python3 -m venv Auto-GINClus_venv
source Auto-GINClus_venv/bin/activate  
```

### Install PyMOL (optional - required to generate images):  
```
sudo apt-get install -y pymol
```

If running Auto-GINClus inside virtual environment then use the following command to add the path of pymol to the virtual environment path:

```
python3 -m venv Auto-GINClus_venv --system-site-packages
```

PyMOL can also be installed directly by downloading the OS-specific version from https://pymol.org/. However, the open-source PyMOL can be obtained by compiling the source code(Python 3.6+ is required). The steps to compile the PyMOL source code is described in the [README-PyMOL-build](README-PyMOL-build.md) file.


## Run Instructions

### \# Run commands for sample input file:
The sample input file [Unknown_motif_location_IL_input_PDB_sample.csv](data/Unknown_motif_location_IL_input_PDB_sample.csv) contains the locations of some RNA loop regions.

**_1. Sample input file run command for subcluster output generation:_**
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB_sample.csv' -o 'output/' -idt pdb -d web -e 0 -w 0 -val 0.064 -test 0.063 -t -k 6 -sd 3.5
```
**_2. Sample input file run command for Q-score and new RSM generation:_**
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB_sample.csv' -o 'output/' -idt pdb -d web -e 0 -w 0 -val 0.064 -test 0.063 -t -k 6 -sd 3.5 -q -qt 0.5 -R 5
```
**_3. Sample input file run command for PyMOL image generation_**
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB_sample.csv' -o 'output/' -idt pdb -d web -e 0 -w 0 -val 0.064 -test 0.063 -t -k 6 -sd 3.5 -p
```
**_4. Sample input file run command for Training_**
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB_sample.csv' -o 'output/' -idt pdb -d web -e 0 -w 0 -val 0.064 -test 0.063 -k 6 -sd 3.5
```
*** Note: PyMOL must be installed first in order to generate images

### \# Overall run commands:
      
**_Run command:_** python3 run.py [-i1 'Train_motif_location_IL_PDB_input.csv'] [-i2 'Unknown_motif_location_IL_PDB_input.csv'] [-o 'output/'] [-e 0] [-d web] [-idt pdb] [-t True] [-w 1] [-val 0.064] [-test 0.063] [-f True] [-c True] [-sc True] [-k 2000] [-s False] [-kmin 200] [-kmax 2500] [-kinc 50] [-q False] [-qt 5] [-r 5] [-p False] [-clean False]  
**_Help command:_** python3 run.py -h  
**_Optional arguments:_** 
```
  -h, --help    	Show this help message and exit
  -i1 [I1]      	Input file containing training motif locations. Default:'Train_motif_location_IL_input_PDB.csv'.
  -i2 [I2]      	Input file containing RNA loop locations. Default:'Unknown_motif_location_IL_input_PDB.csv'.
  -o [O]        	Path to the output files. Default: 'output/'.
  -e [E]        	Number of extended residues beyond loop boundary to generate the loop.cif file. Default: 0.
  -d [D]        	Use 'tool' to generate annotation from DSSR tool, else use 'web' to generate annotation from DSSR website. Default: 'web'.
  -idt [IDT]    	Use 'fasta' if input motif index type is FASTA, else use 'pdb' if input motif index type is PDB. Default: 'pdb'.
  -t [T]        	Trains the model if t = True, else uses the previously trained model weight. To set the parameter to False use '-t'. Default: True.
  -w [W]        	Use '1' to save the new model weight, otherwise, use '0'. Default: '1'.
  -val [VAL]    	Set the percentage of validation data. Default: '0.064'.
  -test [TEST]  	Set the percentage of test data. Default: '0.063'.
  -f [F]        	Generates features for unknown motifs if True, else uses the previously generated features. To set the parameter to False use '-f'. Default: True.
  -c [C]        	Generates cluster output if True, else uses the previously generated clustering output. To set the parameter to False use '-c'. Default: True.
  -sc [SC]      	Generates subcluster output if True, else uses the previously generated subcluster output. To set the parameter to False use '-sc'. Default: True.
  -k [K]        	Define the number of clusters (value of K) to be generated. Default: 2000.
  -s [S]        	Generates the optimal value of K by calculating silhouette score and SMCR for different number of clusters between Kmax and Kmin. Default: False.
  -kmin [KMIN]  	The minimum number of clusters for which the silhoutte score and SMCR will be calculated. Default: 200.
  -kmax [KMAX]  	The maximum number of clusters for which the silhoutte score and SMCR will be calculated. Default: 2500.
  -kinc [KINC]  	The increase in number of clusters at each step (starting from Kmin and upto Kmax) while calculating silhoutte score and SMCR. Default: 50.
  -sd [SD]          Define the distance threshold for subclustering. Default: 3.
  -q [Q]        	If True, generates Q-score for output subclusteres. Default: False.
  -qt [QT]  	    Define the value of Q-score threshold Q for identifying new RNA structural motifs. Default: 5.
  -r [R]  	        Define the value of recurrence threshold R for identifying new RNA structural motifs. Default: 5.
  -p [P]        	If True, generates PyMOL images for output subclusteres. Default: False.
  -clean [CLEAN]    If True, cleans the data folder. Recommended to use before running the tool for new input dataset, otherwise data conflict might arise. Default: False.
  -pickle [PICKLE]	If True, reads previously generated alignment data from pickle files. Default: False.
```

**_Input:_** Auto-GINClus takes the locations of RSMs and RNA loop regions as input from two separate input files inside the [data](data/) folder. These locations are expected to be in PDB index, but it can be changed into FASTA index by setting the "-idt" parameter to "fasta".    

Here, the RSM locations (internal loop motif locations [Train_motif_location_IL_input_PDB.csv](data/Train_motif_location_IL_input_PDB.csv), hairpin loop motif locations [Train_motif_location_HL_input_PDB.csv](data/Train_motif_location_HL_input_PDB.csv)) have been collected from previous works (see 'Data Collection' section in the manuscript). RNA loop locations (internal loop locations [Unknown_motif_location_IL_input_PDB.csv](data/Unknown_motif_location_IL_input_PDB.csv), hairpin loop locations [Unknown_motif_location_HL_input_PDB.csv](data/Unknown_motif_location_HL_input_PDB.csv)) have been collected from RNA-NRD V3 dataset.
1. __Train_motif_location_input:__ contains the locations of RSMs used for training. These locations can be collected from PDB files. Each line in the input file starts with the family name, followed by RNA motif locations. The motif locations are provided using the format 'PDBID_CHAIN:locations'. Example motif location: '4LCK_B:66-71', '1U9S_A:61-64_86-87', '4RGE_C:10-13_24-25_40-43'. Example input files: 'Train_motif_location_IL_input_PDB.csv', 'Train_motif_location_HL_input_PDB.csv'. _Note: Auto-GINClus can also be trained with unknown RNA loop regions by providing locations of loop regions in the training input file (it's self-supervised and doesn't depend on data label/class)._
2. __Unknown_loop_location_input:__ contains the locations of RNA loop regions. Uses the similar format as the file Train_motif_location_input files. Example input files: 'Unknown_motif_location_IL_input_PDB.csv', 'Unknown_motif_location_HL_input_PDB.csv'.


**_Output:_** Generates the following output files inside the user defined output folder, default: 'output'.
1. __Motif_candidate_embeddings.tsv:__ contains the embeddings generated for each RNA loop. These embeddings are used to cluster the RNA loop regions.
2. __Cluster_output.csv:__ contains the clustering output of RNA loops generated by K-means clustering algorithm.
3. __Subcluster_output.csv:__ contains the subclustering output of RNA loops generated by Hierachical Agglomerative clustering algorithm.
4. __Subcluster_output_qscore.csv:__ contains the Q-score (quality score) generated for each subcluster. Higher value of Q-score indicates better quality for a subcluster.
4. __New_RNA_motif_subclusters.csv:__ contains the subclusters with new RSMs. Generated based on Q and R thresholds.
5. __subcluster_images folder:__ contanis the images generated for each subcluster. The images will be generated if PyMOL is installed and "-p" parameter is used while running Auto-GINClus.


**_Example run commands:_**
1. __For internal loops:__ 
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -o 'output/' -idt pdb -d web -e 0 -w 1 -val 0.064 -test 0.063 -k 2400 -sd 2.5
```
2. __For hairpin loops:__ 
```
python3 run.py -i1 'Train_motif_location_HL_input_PDB.csv' -i2 'Unknown_motif_location_HL_input_PDB.csv' -o 'output/' -idt pdb -d web -e 0 -w 1 -val 0.064 -test 0.063 -k 1650 -sd 2.5
```
3. __For automatically calculating K value:__ 
``` 
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -o 'output/' -idt pdb -d web -e 0 -w 1 -val 0.064 -test 0.063 -s True -kmax 300 -kmin 600 -kinc 100
```
4. __For image generation:__ 
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -d web -p
```
5. __For Q-score generation:__ 
```
python3 run.py -i1 'Train_motif_location_IL_input_PDB.csv' -i2 'Unknown_motif_location_IL_input_PDB.csv' -d web -q
```


**_Output files generated by Auto-GINClus for the RNA loops collected from RNA-NRD Version 3:_**  
The outputs generated by Auto-GINClus for loops collected from RNA-NRD Version 3 is provided inside [output/](output/) folder. For internal loops, the output files are inside folder [output/Internal_Loop_Output/](output/Internal_Loop_Output) and for hairpin loops, the output files are inside folder [output/Hairpin_Loop_Output/](output/Hairpin_Loop_Output).
    
### Important Notes   
*** The subcluster images and superimposed images of new RNA structurals motifs for [internal loops](output/Internal_Loop_Output/New_RNA_Motif_Subcluster_Images/) and [hairpin loops](output/Hairpin_Loop_Output/New_RNA_Motif_Subcluster_Images/) are provided inside folder [output](output/). Due to space limitation, we didn't provide the images for all the subclusters (3525 internal loop and 2384 hairpin loop subclusters).  
*** The subclusters images for the seven novel internal loop motif families are provided inside folder [New_Motif_Family_Images](output/Internal_Loop_Output/New_Motif_Family_Images/).  
*** The value of K (number of clusters) can be generated automatically by setting the parameter -s to True. While automatically calculating the value of K, please make sure to also set the parameters -kmin, -kmax and -kinc accordingly.   
*** Generating images for subclusters is optional and it takes comparatively longer to generate all the images.   
*** Needs to install/download PyMOL to generate the subcluster images using Auto-GINClus.  
*** Everytime before running Auto-GINClus for new input datasets, it is recommended to use -clear commad to clean the data directory, this will avoid any data conflicts between existing and new data files.  
*** Auto-GINClus can be run using both CPU and GPU. Use GPU to run Auto-GINClus faster. To run using GPU, load cuda (cuda-12.6.0 or any version compatible with your pyrotch version).  
*** By default, Auto-GINClus downloads DSSR annotations from DSSR web-server. In order to generate annotations using DSSR tool, save the tool inside [DSSR](src/my_lib/DSSR/) folder.  

### Contact
For any questions, please contact nabila.shahnaz.khan@ucf.edu

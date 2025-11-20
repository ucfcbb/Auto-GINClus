qscore_threshold = 0.49
R_val = 5

qscore_file = 'HL_Subcluster_output_qscore.csv'
subcluster_file = 'HL_Subcluster_output.csv'
output_file = 'HL_New_RNA_motif_subclusters.csv'

### Filtering subclusters based on Q-score threshold and R
Filtered_subcluster_list = []
fin1 = open(qscore_file, "r")
line = fin1.readline()

while(True):
    line = fin1.readline()
    if line == "":
        break

    subclus, qscore, no_motifs = line.strip("\n").split(",")

    if float(qscore) > qscore_threshold and int(no_motifs) >= R_val:
        Filtered_subcluster_list.append(subclus)

fin1.close()

print(Filtered_subcluster_list)

### Writing list of subclusters containing new RNA motifs
fout = open(output_file, "w")
fout.write("Motif_location (PDB)\tCluster_id\tSubcluster_id\tFamily_label\n")

fin2 = open(subcluster_file, "r")
line = fin2.readline() 

while(True):
    line = fin2.readline()
    if line == "":
        break

    PDB_location, cluster_id, subcluster_id, family = line.split("\t")
    if subcluster_id in Filtered_subcluster_list:
        fout.write(line)

fin2.close()
fout.close()

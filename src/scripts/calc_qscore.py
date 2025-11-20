import os
import sys
import copy
import networkx as nx
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt



def read_max_min_val(input_file_location):

    MAX_SC_Score = 0
    MAX_SC_RMSD = 0
    MAX_SC_AL = 0
    MAX_TM_Score = 0
    MAX_TM_RMSD = 0
    MAX_TM_AL = 0
    
    infile = open(input_file_location, "r")
    header = infile.readline()

    while(True):
        line = infile.readline()
        if line == "":
            break

        line = line.split(",")
        subclus, pdb1, pdb2, SC_score, SC_RMSD, SC_AL, TM_score, TM_RMSD, TM_AL = line[0], line[1], line[2], float(line[3]), float(line[4]), float(line[5]), float(line[8]), float(line[9]), float(line[10])

        if SC_score > MAX_SC_Score: MAX_SC_Score = SC_score
        if SC_RMSD > MAX_SC_RMSD: MAX_SC_RMSD = SC_RMSD
        if SC_AL > MAX_SC_AL: MAX_SC_AL = SC_AL
        if TM_score > MAX_TM_Score: MAX_TM_Score = TM_score
        if TM_RMSD > MAX_TM_RMSD: MAX_TM_RMSD = TM_RMSD
        if TM_AL > MAX_TM_AL: MAX_TM_AL = TM_AL

    return MAX_SC_Score, MAX_SC_RMSD, MAX_SC_AL, MAX_TM_Score, MAX_TM_RMSD, MAX_TM_AL



def calc_qscore_for_each_subcluster(SUBCLUSTER_NO):

    # Nabila debug
    ### Count number of loops in a cluster
    SUBCLUS_member_dic = {}
    input_subcluster_out = 'output/temp/Subcluster_output.in'
    subclus_file = open(input_subcluster_out, "r")

    while(True):
        line = subclus_file.readline()
        if line == "":
            break
        line = line.strip("\n").split(",")
        subclus_no = int(line[0].split("_")[1])
        SUBCLUS_member_dic[subclus_no] = len(line) - 1
    subclus_file.close()


    SUBCLUS_Dic = {}
    SUBCLUS_motif_dic = {}
    
    input_file_location = 'output/temp/Subcluster_output_qscore_temp.csv'
    output_file_location = 'output/Subcluster_output_qscore.csv'

    infile = open(input_file_location, "r")
    header = infile.readline()

    MAX_SC_Score, MAX_SC_RMSD, MAX_SC_AL, MAX_TM_Score, MAX_TM_RMSD, MAX_TM_AL = read_max_min_val(input_file_location)

    while(True):
        line = infile.readline()
        if line == "":
            break

        line = line.split(",")
        subclus, pdb1, pdb2, SC_score, SC_RMSD, SC_AL, TM_score, TM_RMSD, TM_AL = line[0], line[1], line[2], float(line[3]), float(line[4]), float(line[5]), float(line[8]), float(line[9]), float(line[10])

        ### Normalizing values
        SC_score_norm = (SC_score - 0)/(MAX_SC_Score - 0)
        SC_RMSD_norm = (SC_RMSD - 0)/(MAX_SC_RMSD - 0)
        SC_AL_norm = (SC_AL - 0)/(MAX_SC_AL - 0)
        TM_score_norm = TM_score
        TM_RMSD_norm = (TM_RMSD - 0)/(MAX_TM_RMSD - 0)
        TM_AL_norm = (TM_AL - 0)/(MAX_TM_AL - 0)
        
        subclus_no = int(subclus.split("_")[1])

        if subclus_no not in SUBCLUS_Dic:
            SUBCLUS_Dic[subclus_no] = [0, 0, 0, 0, 0, 0, 0]
            SUBCLUS_motif_dic[subclus_no] = []
       
        SUBCLUS_Dic[subclus_no][0] = SUBCLUS_Dic[subclus_no][0] + SC_score_norm
        SUBCLUS_Dic[subclus_no][1] = SUBCLUS_Dic[subclus_no][1] + SC_RMSD_norm
        SUBCLUS_Dic[subclus_no][2] = SUBCLUS_Dic[subclus_no][2] + SC_AL_norm
        SUBCLUS_Dic[subclus_no][3] = SUBCLUS_Dic[subclus_no][3] + TM_score_norm
        SUBCLUS_Dic[subclus_no][4] = SUBCLUS_Dic[subclus_no][4] + TM_RMSD_norm
        SUBCLUS_Dic[subclus_no][5] = SUBCLUS_Dic[subclus_no][5] + TM_AL_norm
        SUBCLUS_Dic[subclus_no][6] = SUBCLUS_Dic[subclus_no][6] + 1

        if pdb1 not in SUBCLUS_motif_dic[subclus_no]:
            SUBCLUS_motif_dic[subclus_no].append(pdb1)

    Cluster_qscore = []

    MAX_Cluster_size = 0
    for key in SUBCLUS_motif_dic:
        cur_len = len(SUBCLUS_motif_dic[key])
        if cur_len > MAX_Cluster_size: MAX_Cluster_size = cur_len

    for key in SUBCLUS_Dic:

        if SUBCLUS_Dic[key][6] != 0:
            AVG_SC_SCORE = SUBCLUS_Dic[key][0] / SUBCLUS_Dic[key][6]
            AVG_SC_RMSD = SUBCLUS_Dic[key][1] / SUBCLUS_Dic[key][6]
            AVG_SC_AL = SUBCLUS_Dic[key][2] / SUBCLUS_Dic[key][6]
            AVG_TM_SCORE = SUBCLUS_Dic[key][3] / SUBCLUS_Dic[key][6]
            AVG_TM_RMSD = SUBCLUS_Dic[key][4] / SUBCLUS_Dic[key][6]
            AVG_TM_AL = SUBCLUS_Dic[key][5] / SUBCLUS_Dic[key][6]
            NO_motifs = len(SUBCLUS_motif_dic[key])
            NO_motifs_norm = NO_motifs / MAX_Cluster_size 

            ### Calculate Q-score:
            # Scanx_Q_score = AVG_SC_SCORE + AVG_SC_AL + NO_motifs_norm - AVG_SC_RMSD
            # TM_Q_score = AVG_TM_SCORE + AVG_TM_AL + NO_motifs_norm - AVG_TM_RMSD
            # Q_score = (Scanx_Q_score + TM_Q_score)/2
            # Cluster_qscore.append((key, Q_score))

            ### Calculate Q-score:
            # Nabila debug
            Scanx_Q_score = AVG_SC_SCORE + AVG_SC_AL - AVG_SC_RMSD
            TM_Q_score = AVG_TM_SCORE + AVG_TM_AL - AVG_TM_RMSD
            Q_score = (Scanx_Q_score + TM_Q_score)/2
            
            # Nabila debug
            number_of_motifs = SUBCLUS_member_dic[key]
            # Cluster_qscore.append((key, Q_score))
            Cluster_qscore.append((key, Q_score, number_of_motifs))

    sorted_cluster_qscore = sorted(Cluster_qscore, key=lambda x: x[1], reverse=True)
  
    infile.close()

    ### Write output in a CSV file
    outfile = open(output_file_location, "w")

    # Nabila debug
    # outfile.write("Subcluster_id,Q-score\n")
    outfile.write("Subcluster_id,Q-score,No_of_motifs\n")

    for clus in sorted_cluster_qscore:
       
        subclus = clus[0]
        qscore = clus[1]
        # Nabila debug
        number_of_motifs = clus[2]
        # outfile.write("%s,%s\n" % (subclus, qscore))
        outfile.write("%s,%s,%s\n" % (subclus, qscore, number_of_motifs))
        
    outfile.close()


def identify_new_motifs(qscore_threshold, R_val):

    qscore_file = 'output/Subcluster_output_qscore.csv'
    subcluster_file = 'output/Subcluster_output.csv'
    output_file = 'output/New_RNA_motif_subclusters.csv'

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



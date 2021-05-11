import shutil
import os

points_list = [12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240]

for points in points_list:

    #directory = '/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/analysis//points={}'.format(points) + '/learning curves'
    #os.makedirs(directory)
    for run in range(1, 11):

        source = '/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/new_upsampled/points={}'.format(points) \
           +'/Run={}'.format(run) + '/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True/rep1/iter_gbest_reached.csv'

        dest = '/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/analysis/points={}'.format(points) + '/learning curves/iter_gbest_reached_run_{}'.format(run) +'.csv'


        shutil.copy(source, dest)
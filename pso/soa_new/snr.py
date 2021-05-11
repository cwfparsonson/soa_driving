import numpy as np
import matplotlib.pyplot as plt

'''IGNORE THIS
file_path = './snr_outputs'
file_names = ['aco_snr.csv','pso_snr.csv','ga_snr.csv']
outputs = [np.genfromtxt('{}/{}'.format(file_path,file_name))[1:-135] for file_name in file_names]
'''

outputs = []
#
outputs.append(np.genfromtxt('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/n=160/Run=2/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True/rep1/optimised_PV.csv'))

#/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/n=160/Run=2/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True/rep1/optimised_PV.csv
#/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/n=160/Run=2/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True/rep1/initial_PV.csv'))
#/Users/hadi/Desktop/SOA_Complexity/PSO_Data/n=160_iter=150/Run=1/points=240/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True/rep1/initial_PV.csv
#example SNR

ss_idx = int(len(outputs[0])*0.05) #will be 12 for 240 point signal
#print(len(outputs[0]))
signal_ss = []

for output in outputs:
    #get the index at which the signal initially enters the 5% region

    ss = np.mean(output[-ss_idx:])     #get mean of last 12 values of signal
    #print(output[-ss_idx:]) 
    ss_low = ss - 0.23*ss           

    #print(np.abs(output - ss_low))
    #output = np.ma.masked_equal(output,0.0)

    begin_idx = np.argmax(np.abs(output - ss_low))

    #trim signal to include only region after enters 5%
    output = output[begin_idx:]

    print(output[begin_idx:])

    plt.figure(figsize=(12,6))
    plt.plot(output)
    plt.plot([ss]*len(output))

    #print the variance of the signal from when it enters 5% region until the end
    print(np.var(output))

    #get noise from this region and calculate SNR (dB units)
    rms_noise = np.sqrt((np.mean(np.power(output-ss,2))))
    signal_ss.append((output,ss,rms_noise))
    snr = 10*np.log10(np.power(ss/rms_noise,2))
    print("SNR = {}".format(snr))


plt.savefig('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/snr.png') 
plt.show()

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

ft = []
st = []
os = []
us = []

def multiplyList(myList) :
     
    # Multiply elements one by one
    #result = 1
    newList = [x * 100 for x in myList]
    return newList

if __name__ == '__main__':

    for run in range(1,11,1):

        directory = "/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/n=160/Run={}".format(run) + "/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True" \
                 "/rep1/ft_rt_st_os_analysis.csv"
        
        
        y = pd.read_csv(directory, header=None)

        ft.append(y[0].iloc[-1])
        st.append(y[1].iloc[-1]) 
        os.append(y[4].iloc[-1])  
        us.append(y[2].iloc[-1])
    
    os = multiplyList(os)
    us = multiplyList(us)
    
    print(ft, st, os, us)
   
   
    plt.figure(figsize=(12,6))

    

    plt.title('Fall: n = 160, iter = 150')
    plt.xlabel('Settling time (s)')
    plt.ylabel('Fall time (s)')
    plt.xlim(10e-10, 30e-10)
    plt.ylim(1e-10, 9e-10)

    col = ['black', 'blue', 'green', 'magenta', 'purple', 'darkgrey', 'lightcoral', 'olive', 'cyan', 'orange']
    for i in range(10):
        plt.scatter(st[i], ft[i], c=col[i])

    plt.legend(['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8', 'Run 9', 'Run 10'], loc='best')

    plt.savefig('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/ft_st.png')  
 
 
    plt.figure(figsize=(12,6))
    plt.title('Fall: n = 160, iter = 150')
    plt.xlabel('Undershoot (%)')
    plt.ylabel('Overshoot (%)')
    plt.xlim(92, 100)
    plt.ylim(5, 24)


    for i in range(10):
        plt.scatter(us[i], os[i], c=col[i])

    plt.legend(['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8', 'Run 9', 'Run 10'], loc='best')

    plt.savefig('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Fall/New_23_04/os_us.png')  
 

    plt.show()


    
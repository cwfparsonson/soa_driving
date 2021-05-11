import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

points_list = [30, 40, 48, 60, 80, 120, 240]

rt_240 = []
st_240 = []
os_240 = []
rt_120 = []
st_120 = []
os_120 = []
rt_80 = []
st_80 = []
os_80 = []
rt_60 = []
st_60 = []
os_60 = []
rt_48 = []
st_48 = []
os_48 = []
rt_40 = []
st_40 = []
os_40 = []
rt_30 = []
st_30 = []
os_30 = []
x = []

def multiplyList(myList) :
     
    # Multiply elements one by one
    #result = 1
    newList = [x * 100 for x in myList]
    return newList

if __name__ == '__main__':
    for points in points_list:
        for run in range(1,11,1):
            #directory = "/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/analysis/points=240/learning curves/MSE_Iter_240.csv"
            #directory = "/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/analysis/points={}".format(points) + "/rt_st_os_analysis/run_{}".format(run)+".csv"
            directory = "/Users/hadi/Desktop/SOA_Complexity/PSO_Data/480 upsampling/ParameterSweep/n=160/points={}".format(points) + "/Run={}".format(run) + "/test_0/data/n_160_mxvl_1.0_mnvl_-1.0_ivf_0.05_mvf_0.05_irmx_150_rm_1_cstf_mSE_sif_None_wi_0.9_wf_0.5_c1_0.2_c2_0.2_aptacl_True_offsf_0.2_onsf_2.0_emb_True" \
                 "/rep1/rt_st_os_analysis.csv"
            
            """
            for i in range(1,10,1):
                x = pd.read_csv(directory, usecols= [2*i - 2])
                y = pd.read_csv(directory, usecols= [2*i - 1]) 
                i = i+1
            """
            y = pd.read_csv(directory, header=None)
            #x = pd.read_csv(directory, usecols= [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
            #y = pd.read_csv(directory, usecols= [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) 

            if points == 30:
                rt_30.append(y[0].iloc[-1])
                st_30.append(y[1].iloc[-1]) 
                os_30.append(y[2].iloc[-1])  
                x_30 = [30]*10

            elif points == 40:
                rt_40.append(y[0].iloc[-1])
                st_40.append(y[1].iloc[-1])
                os_40.append(y[2].iloc[-1]) 
                x_40 = [40]*10

            elif points == 48:
                rt_48.append(y[0].iloc[-1])
                st_48.append(y[1].iloc[-1])
                os_48.append(y[2].iloc[-1]) 
                x_48 = [48]*10

            elif points == 60:
                rt_60.append(y[0].iloc[-1])
                st_60.append(y[1].iloc[-1])
                os_60.append(y[2].iloc[-1]) 
                
                x_60 = [60]*10

            elif points == 80:
                rt_80.append(y[0].iloc[-1])
                st_80.append(y[1].iloc[-1])
                os_80.append(y[2].iloc[-1]) 
                x_80 = [80]*10

            elif points == 120:
                rt_120.append(y[0].iloc[-1])
                st_120.append(y[1].iloc[-1])
                os_120.append(y[2].iloc[-1]) 
                
                x_120 = [120]*10

            elif points == 240:
                rt_240.append(y[0].iloc[-1])
                st_240.append(y[1].iloc[-1])
                os_240.append(y[2].iloc[-1]) 
                x_240 = [240]*10

            #print(y.iat[-1])
            #print(mse)
            
            #print(x)
            #print(y[run])
            x.append(points)

            
        #print(rt)
        #print(st)    
        #y[~numpy.isnan(y)]

        #y = list(filter(None, y))
        #print(x[run])
            #x = points 
    
    #print(st_30)
    
    os_30 = multiplyList(os_30)
    os_40 = multiplyList(os_40)
    os_48 = multiplyList(os_48)
    os_60 = multiplyList(os_60)
    os_80 = multiplyList(os_80)
    os_120 = multiplyList(os_120)
    os_240 = multiplyList(os_240)
   
    plt.figure()
    plt.figure(figsize=(12,6))

    plt.title('upsampling = 480')
    plt.xlabel('Settling time (s)')
    plt.ylabel('Rise time (s)')
    plt.xlim(1e-10, 20e-10)
    plt.ylim(1e-10, 10e-10)

    plt.scatter(st_30, rt_30, c='blue')
    plt.scatter(st_40, rt_40, c='purple')
    plt.scatter(st_48, rt_48, c='green')
    plt.scatter(st_60, rt_60, c='magenta')
    plt.scatter(st_80, rt_80, c ='orange')
    plt.scatter(st_120, rt_120, c= 'red')
    plt.scatter(st_240, rt_240, c= 'black')
    plt.legend(['30 points', '40 points', '48 points', '60 points', '80 points', '120 points', '240 points'], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.,fontsize='xx-small')
    plt.savefig('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/480 upsampling/n=160 analysis/rt_st.png')

    plt.close()
    
    plt.figure(2)
    plt.figure(figsize=(12,6))

    plt.title('upsampling = 480')
    plt.ylabel('overshoot %')
    plt.xlabel('number of points')
    plt.xlim(0, 250)
    plt.ylim(0, 10)
    
    plt.scatter(x_30, os_30, c='blue')
    plt.scatter(x_40, os_40, c='purple')
    plt.scatter(x_48, os_48, c='green')
    plt.scatter(x_60, os_60, c='magenta')
    plt.scatter(x_80, os_80, c ='orange')
    plt.scatter(x_120, os_120, c= 'red')
    plt.scatter(x_240, os_240, c= 'black')

    plt.legend(['30 points', '40 points', '48 points', '60 points', '80 points', '120 points', '240 points'], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.,fontsize='xx-small')
    plt.savefig('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/480 upsampling/n=160 analysis/overshoot.png')
    #lt.yticks(np.arange(min(mse), max(mse)+1, 0.0000000000001))


    #plt.show()

    #plt.close()  
    
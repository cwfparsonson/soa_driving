from scipy import signal
import scipy.interpolate
from soa import subsampling
import matplotlib.pyplot as plt
import numpy as np
import copy
   



def find_x_init(tf):
    """
    This method calculates the state-vector from a long -1 drive signal. 
    Must call before sending / receiving signals to / from transfer function 
    model

    Args:
    - tf = transfer function

    Returns:
    - X0 = system's state-vector result for steady state
    """
    U = np.array([0.5] * 480)
    T = np.linspace(0, 40e-9, 480)

    (_, _, xout) = signal.lsim2(tf, U=U, T=T, X0=None, atol=1e-13)

    
    X0 = xout[-1]
    
    return X0


def getTransferFunctionOutput(tf, U, T, atol=1e-12):
    """
    This method sends a drive signal to a transfer function model and gets the 
    output

    Args:
    - tf = transfer function
    - U = signal to drive transfer function with
    - T = array of time values
    - X0 = initial value
    - atol = scipy ode func parameter

    Returns:
    - PV = resultant output signal of transfer function
    """

    X0 = find_x_init(tf)


    #(_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)
    
    # plt.figure("Before upsampling")
    # plt.title("number of points = {}".format(points) + "\nOutput of SOA: Before upsampling")
    # plt.plot(PV)
    # plt.show()

    # plt.figure("Before upsampling")
    # plt.title("number of points = {}".format(points) + "\nInput of SOA: Before upsampling")
    # plt.plot(U)
    # plt.show()

    p2p = 240/len(U)
    U = np.array(U)
    
    T = np.linspace(0, 20e-9, 240)
    T = np.array(T)
    U = np.repeat(U, p2p)

    # plt.figure("After upsampling")
    # plt.title("number of points = {}".format(points) + "\nInput of SOA: After upsampling")
    # plt.plot(U)
    # plt.show()
    

    (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)


    # print ("Tr: %fs"%(T[next(i for i in range(0,len(PV)-1) if PV[i]>PV[-1]*.90)]-T[0]))

    
    #print(sig)
    #print(labels)


    # tenY = ((PV[-1]-PV[0])*0.1 + PV[0])
    # ninetyY = ((PV[-1]-PV[0])*0.9 + PV[0])


    # interp_func = scipy.interpolate.interp1d(PV[0:800], T[0:800])
    # tenX = interp_func(tenY)
    # ninetyX = interp_func(ninetyY)
    
    # #print("DemFooOnes {}".format(foo))


    # #tenX = np.interp(tenY, PV, T)
    # #ninetyX = np.interp(ninetyY, PV, T)

    # #np.interp(0.1*PV[-1], T, PV)
    # print("this is ten Y {}".format(tenY))
    # print("this is ten X {}".format(tenX))

    # print("this is ninety Y {}".format(ninetyY))
    # print("this is ninety X {}".format(ninetyX))
    
    # RiseT = ninetyX - tenX
    # print("rise time = {}".format(RiseT))

    # print(len(PV))
    # print(len(T))

    
    
    # #print(yidx)

    # plt.plot(T, PV)

    # plt.plot(tenX, tenY,
    # marker='x',
    # markersize=6, 
    # color="black", 
    # label='Ten Index')

    # plt.plot(ninetyX, ninetyY,
    # marker='x',
    # markersize=6, 
    # color="red", 
    # label='Ninety Index')

    # #plt.show()

    min_PV = np.copy(min(PV))
    if min_PV < 0:
        for i in range(0, len(PV)):
            PV[i] = PV[i] + abs(min_PV) # translate signal up

    return PV



# def step_info(t,yout):
#     print ("OS: %f%s"%((yout.max()/yout[-1]-1)*100,'%'))
#     print ("Tr: %fs"%(t[next(i for i in range(0,len(yout)-1) if yout[i]>yout[-1]*.10)]-t[0]))
#     #print ("Ts: %fs"%(t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1])>1.02)]-t[0]))





def plot_output(signals=[], labels=[]):
    plt.figure("after upsampling")
    #plt.title("number of points = {}".format(points) + "\nOutput of SOA: after upsampling by {}".format(2400/points) + "x")
    for sig, lab in zip(signals, labels):
        plt.plot(sig, label=str(lab))


    #plt.legend(loc='upper left')
    #plt.savefig("/Users/hadi/Desktop/2400 upsampled/number of points = {}".format(points) + ".png")
    #plt.show()


def gen_dummy_tf_num_den(num_fac, a_facs=[]):
    num = [2.01199757841099e85]
    den = [
        1.64898505756825e0,
        4.56217233166632e10,
        3.04864287973918e21,
        4.76302109455371e31,
        1.70110870487715e42,
        1.36694076792557e52,
        2.81558045148153e62,
        9.16930673102975e71,
        1.68628748250276e81,
        2.40236028415562e90,
    ]

    dummy_num = copy.deepcopy(num)[0]
    dummy_den = copy.deepcopy(den)

    dummy_num *= num_fac
    a_fac = iter(a_facs)
    for idx in reversed(range(10)):
        dummy_den[idx] *= next(a_fac)
    
    return dummy_num, dummy_den


def gen_tfs(num_facs=[1], 
            a0_facs=[1], 
            a1_facs=[1], 
            a2_facs=[1], 
            a3_facs=[1], 
            a4_facs=[1], 
            a5_facs=[1], 
            a6_facs=[1], 
            a7_facs=[1], 
            a8_facs=[1], 
            a9_facs=[1], 
            all_combos=False):
    tfs = []
    labels = []

    if all_combos:
        for num_fac in num_facs:
            for a0_fac in a0_facs:
                for a1_fac in a1_facs:
                    for a2_fac in a2_facs:
                        for a3_fac in a3_facs:
                            for a4_fac in a4_facs:
                                for a5_fac in a5_facs:
                                    for a6_fac in a6_facs:
                                        for a7_fac in a7_facs:
                                            for a8_fac in a8_facs:
                                                for a9_fac in a9_facs:
                                                    dummy_num,dummy_den = gen_dummy_tf_num_den(num_fac, [a0_fac,a1_fac,a2_fac,a3_fac,a4_fac,a5_fac,a6_fac,a7_fac,a8_fac,a9_fac])
                                                    tf = signal.TransferFunction([dummy_num], dummy_den)
                                                    tfs.append(tf)
                                                    labels.append('num={},a={}|{}|{}|{}|{}|{}|{}|{}|{}|{}'.format(num_fac,a0_fac,a1_fac,a2_fac,a3_fac,a4_fac,a5_fac,a6_fac,a7_fac,a8_fac,a9_fac))

    else:
        if num_facs == [1]:
            pass
        else:
            for num_fac in num_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(num_fac, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('num_f={}'.format(num_fac))

        if a0_facs == [1]:
            pass
        else:
            for a0_fac in a0_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [a0_fac, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a0_f={}'.format(a0_fac))
            
        if a1_facs == [1]:
            pass
        else:
            for a1_fac in a1_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, a1_fac, 1, 1, 1, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a1_f={}'.format(a1_fac))

        if a2_facs == [1]:
            pass
        else:
            for a2_fac in a2_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, a2_fac, 1, 1, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a2_f={}'.format(a2_fac))

        if a3_facs == [1]:
            pass
        else:
            for a3_fac in a3_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, a3_fac, 1, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a3_f={}'.format(a3_fac))

        if a4_facs == [1]:
            pass
        else:
            for a4_fac in a4_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, a4_fac, 1, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a4_f={}'.format(a4_fac))
        
        if a5_facs == [1]:
            pass
        else:
            for a5_fac in a5_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, 1, a5_fac, 1, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a5_f={}'.format(a5_fac))

        if a6_facs == [1]:
            pass
        else:
            for a6_fac in a6_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, 1, 1, a6_fac, 1, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a6_f={}'.format(a6_fac))

        if a7_facs == [1]:
            pass
        else:
            for a7_fac in a7_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, 1, 1, 1, a7_fac, 1, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a7_f={}'.format(a7_fac))

        if a8_facs == [1]:
            pass
        else:
            for a8_fac in a8_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, 1, 1, 1, 1, a8_fac, 1])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a8_f={}'.format(a8_fac))

        if a9_facs == [1]:
            pass
        else:
            for a9_fac in a9_facs:
                dummy_num, dummy_den = gen_dummy_tf_num_den(1, [1, 1, 1, 1, 1, 1, 1, 1, 1, a9_fac])
                tf = signal.TransferFunction([dummy_num], dummy_den)
                tfs.append(tf)
                labels.append('a9_f={}'.format(a9_fac))
    
    return tfs, labels









if __name__ == '__main__':
    
    #points_list = [30,40,48,60,80,120,240]
    #for points in points_list:
    points = 240
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,points)
    init_OP = np.zeros(points)
    # for transfer function (low point MUST be -1):
    #init_OP[:int(0.25*points)], init_OP[int(0.25*points):] = -1, 0.5 #original - rise

    

    init_OP[:int(0.25*points)], init_OP[int(0.25*points):] = 0.5, -1 #fall 

    #square input
    #init_OP[:int(0.25*points)],init_OP[int(0.25*points):int(0.75*points)],init_OP[int(0.75*points):] = -1, 0.5, -1 
    

    tfs, labels = gen_tfs(num_facs=[1.2],
                            all_combos=False)

    signals = []
    for tf in tfs:
        signals.append(getTransferFunctionOutput(tf,init_OP,t))

    

    #step_info(t, signals)
    plot_output(signals,labels)



    # print(max(signals))
    # print(signals[-1])
    
    #print ("OS: %f%s"%((max(signals)/signals[-1]-1)*100,'%'))

    print('tfs:\n{}'.format(tfs))
    print('Num tfs: {}'.format(len(tfs)))   





















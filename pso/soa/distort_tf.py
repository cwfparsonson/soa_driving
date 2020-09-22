from scipy import signal
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
    U = np.array([-1.0] * 480)
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
    (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)

    min_PV = np.copy(min(PV))
    if min_PV < 0:
        for i in range(0, len(PV)):
            PV[i] = PV[i] + abs(min_PV) # translate signal up

    return PV


def plot_output(signals=[], labels=[]):
    plt.figure()
    for sig, lab in zip(signals, labels):
        plt.plot(sig, label=str(lab))
    plt.legend(loc='upper left')
    plt.show()


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
    num_points = 240
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,num_points)
    init_OP = np.zeros(num_points)
    # for transfer function (low point MUST be -1):
    init_OP[:int(0.25*num_points)], init_OP[int(0.25*num_points):] = -1, 0.5

    tfs, labels = gen_tfs(num_facs=[1.0,1.2,1.4], 
                          a0_facs=[0.8],
                          a1_facs=[0.7,0.8,1.2],
                          a2_facs=[1.05,1.1,1.2],
                          all_combos=False)

    signals = []
    for tf in tfs:
        signals.append(getTransferFunctionOutput(tf,init_OP,t))
    plot_output(signals,labels)

    print('tfs:\n{}'.format(tfs))
    print('Num tfs: {}'.format(len(tfs)))   





















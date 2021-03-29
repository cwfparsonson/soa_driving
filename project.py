from pso.soa import distort_tf_alt


if __name__ == '__main__':
    from soa import devices, signalprocessing, analyse, distort_tf
    from soa.optimisation import PSO, run_test


    import numpy as np
    import multiprocessing
    import pickle
    from scipy import signal
    import os
    import matplotlib.pyplot as plt


    # set dir to save data
    linux = True
    directory = '/home/zceevva/soa_driving/'
    
    def factorize(num):
        return [n for n in range(1, num + 1) if num % n == 0]


    # init basic params
    '''
    num_points_list = np.array(factorize(240))
    num_points_list = num_points_list[num_points_list >= 10]
    '''
    num_points_list = [48]

    time_start = 0
    time_stop = 20e-9

    # set PSO params
    n = 50
    run = 'CPSOSK'
    iter_max = 100
    rep_max = 1 
    max_v_f = 0.05 
    init_v_f = max_v_f 
    cost_f = 'mSE' 
    w_init = 0.9
    w_final = 0.5
    on_suppress_f = 2.0
    q = 3
    # initial transfer function numerator and denominator coefficients
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
    tf = signal.TransferFunction(num, den)
    tfs, labels = distort_tf.gen_tfs(num_facs=[1.0,1.2,1.4], 
                        a0_facs=[0.8],
                        a1_facs=[0.7,0.8,1.2],
                        a2_facs=[1.05,1.1,1.2],
                        all_combos=False)

    # run PSO tests in parallel with multiprocessing
    pso_objs = multiprocessing.Manager().list()
    jobs = []
    for num_points in num_points_list:
        # make directory for this test
        direc = directory + '/num_points_{}'.format(num_points)
        if os.path.exists(direc) == False:
            os.mkdir(direc)

        # basic params
        t = np.linspace(time_start, time_stop, num_points)

        # define initial drive signal
        init_OP = np.zeros(num_points) # initial drive signal (e.g. a step)
        
        init_OP[:int(0.25*num_points)],init_OP[int(0.25*num_points):] = -1, 0.5


        # get initial output of initial signal and use to generate a target set point
        t2 = np.linspace(time_start, time_stop, 240)
        sp = np.zeros((q, 240))
        for i in range(q):
            sp[i] = analyse.ResponseMeasurements(distort_tf.getTransferFunctionOutput(tfs[i],init_OP,t2), t2).sp.sp
        
        
        init_OP = np.tile(init_OP, q)
        
 
        p = multiprocessing.Process(target=run_test, 
                                    args=(direc, 
                                        tfs[:q], 
                                        t,
                                        run, 
                                        init_OP, 
                                        n, 
                                        iter_max, 
                                        rep_max, 
                                        init_v_f, 
                                        max_v_f,
                                        q,
                                        w_init, 
                                        w_final, 
                                        True, 
                                        'pisic_shape', 
                                        on_suppress_f, 
                                        True, 
                                        None, 
                                        cost_f, 
                                        None, 
                                        False, 
                                        linux,
                                        sp, 
                                        pso_objs,))

        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()

    # pickle PSO objects so can re-load later if needed
    PIK = directory + '/pickle.dat'
    data = pso_objs
    with open(PIK, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    
    '''
    Below is an example implementation of the above PSO code. Note that there 
    are 2 main modes of using this PSO implementation:

    1) Simulation (using a transfer function that simulates SOAs)
    2) Experimental 

    To use the experimental setup, you will need all the same equipment, modules,
    specific GPIB addresses etc. that were used in the Connet lab in UCL's
    EEE Robert's building (contact the Optical Networks Group for more info).
    Users outside of ONG will need to write code to interface with their
    own equipment.

    To use the simulation (i.e. the transfer function), the user should not need
    to write any code themselves. Simply changing the below 'directory' variable
    to point this programme to where to store data should be sufficient. 

    The below code runs a PSO simulation, where PSO is optimising 10 different
    SOA transfer functions in parallel. Users can play around with the PSO
    hyperparameters to control PSO performance, convergence properties, 
    run time etc., and can also distort the transfer function by adjusting 
    the distortion coefficients or even implement their own transfer functions
    to simulate their custom SOAs. By optimising different transfer functions,
    users will be able to see how well PSO is generalising to different SOAs.

    While this code is not the 'cleanest', we have tried to insert clear comments
    so that a user wishing to delve deeper into the use of this PSO implementation 
    (beyond running simple transfer function simulations) can follow the logic 
    and implement the same functionality in their own programmes. 

    As a general rule-of-thumb, increasing n (the number of particles) is a 
    reliable way to improve PSO performance and find more optimal solutions.
    '''
    
    
    import soa

    # specify whether running simulation or experiment
    sim = True

    # specify if using linux (or mac) (for backslash or forward slash dirs)
    linux = True

    m = 10
    points = int(m)
    max_points = 40
  
   
    time_start = 0.0
    time_stop = 20e-9 
   

    # config pso params
    n = 3  #number of input driving signals 
    iter_max = 150
    rep_max = 1  
    max_v_f = 0.05
    init_v_f = max_v_f
    cost_f = 'mSE'
    st_importance_factor = None
    w_init = 0.9
    w_final = 0.5
    on_suppress_f = 2.0

    if sim == True:
        # define transfer function(s)
        # N.B. init_OP low must be -1 for tf
        
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

        # set up simulation(s) you want to run
        tfs, _ = distort_tf.gen_tfs(num_facs=[1.0,1.2,1.4], 
                                    a0_facs=[0.8],
                                    a1_facs=[0.7,0.8,1.2],
                                    a2_facs=[1.05,1.1,1.2],
                                    all_combos=False)

        pso_objs = multiprocessing.Manager().list()
        jobs = []
        test_nums = [test+1 for test in range(len(tfs))]
    
        while points <= max_points:    
            counter = 0
            directory = ('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/M={}'.format(points))
            os.mkdir(directory)
            for tf in tfs:
            
                init_OP = np.zeros(points)
                init_OP[:int(0.25*points)],init_OP[int(0.25*points):] = -1, 0.5
                t = np.linspace(time_start,time_stop,points)
                init_PV = distort_tf.getTransferFunctionOutput(tf,init_OP,t)
                sp = analyse.ResponseMeasurements(init_PV, t).sp.sp
                   # set dir to save data
                
                direc = directory + '/test_{}'.format(counter) 
                
                if os.path.exists(direc) == False:
                    os.mkdir(direc)
                p = multiprocessing.Process(target=run_test, 
                                            args=(direc, 
                                              tf, 
                                              t, 
                                              init_OP, 
                                              n, 
                                              iter_max, 
                                              rep_max, 
                                              init_v_f, 
                                              max_v_f, 
                                              w_init, 
                                              w_final, 
                                              True, 
                                              'pisic_shape', 
                                              on_suppress_f, 
                                              True, 
                                              None, 
                                              cost_f, 
                                              st_importance_factor, 
                                              True, 
                                              linux,
                                              sp, 
                                              pso_objs,))
            
                jobs.append(p)
                p.start()

                """
                    # plot composite graph
                    pso_objs = list(pso_objs)
                    plt.figure()
                    plt.plot(t, sp, color='green')
                    for pso_obj in pso_objs:
                        plt.plot(t, pso_obj.gbest_PV)
                    #plt.show()


                    # pickle data
                    PIK = directory + '/pickle.dat'
                    data = pso_objs
                    with open(PIK, 'wb') as f:
                        pickle.dump(data, f)
                """
                counter +=1
            points +=10
        for job in jobs:
            job.join()        
        
    else:
        # set up experiment(s) you want to run
        directory = r"C:\Users\onglab\Desktop\SOA_project\Chris\pso_no_fall_test_09012020" 
        awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
        osc = devices.Agilent86100C("GPIB1::7::INSTR")
        osc.set_acquire(average=True, count=30, points=num_points)
        osc.set_timebase(position=4.2e-8, range_=time_stop-time_start)

        init_OP[:60], init_OP[60:] = -0.5, 0.5 

        psoObject = pso(t, 
                        init_OP, 
                        n, 
                        iter_max, 
                        rep_max, 
                        init_v_f, 
                        max_v_f, 
                        w_init=w_init, 
                        w_final=w_final, 
                        adapt_accel=True, 
                        areas_to_suppress='start_centre', 
                        on_suppress_f=on_suppress_f, 
                        embed_init_signal=True, 
                        path_to_embedded_signal=None, 
                        directory=directory, 
                        cost_f=cost_f, 
                        st_importance_factor=st_importance_factor, 
                        awg=awg, 
                        osc=osc) 
        


    
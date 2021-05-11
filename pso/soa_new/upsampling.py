def __getTransferFunctionOutput(self, tf, U, T, X0, atol=1e-12):
        """
        This method sends a drive signal to a transfer function model and gets 
        the output

        Args:
        - tf = transfer function
        - U = signal to drive transfer function with
        - T = array of time values
        - X0 = initial value
        - atol = scipy ode func parameter

        Returns:
        - PV = resultant output signal of transfer function
        """
        #start_t = time.time()
        

        p2p = self.upsampled/len(U)
        U = np.array(U)
        
        T = np.linspace(0, 20e-9, self.upsampled)
        T = np.array(T)
        U = np.repeat(U, p2p)



       (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)
    
        

        # ensure lower point of signal >=0 (can occur for sims), otherwise
        # will destroy st, os and rt analysis
        min_PV = np.copy(min(PV))
        if min_PV < 0:
            for i in range(0, len(PV)):
                PV[i] = PV[i] + abs(min_PV)

        return PV
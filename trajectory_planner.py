from augmented_mp import AugmentedMathematicalProgram
import numpy as np
import time

def bold(x):
    return '\033[1m'+ x + '\033[0m'


class BoatConfigurationPlanning(object):

    def __init__(self, boat):
        self.boat = boat 
                
    def compute_trajectory(self, minimum_time, maximum_time, state_initial, state_final, time_cost=True, input_cost=False, dif_states_cost=False, states_initialization=None, step_initialization=None,N=100, slack=1e-3):
        
        start = time.time()
        
        #Check variable dimensions
        ##########################
        
        assert state_initial.shape==state_final.shape   # Same number of boats, state vars for final and initial states
        
        #Initialize Optimization Variables
        ##################################       
        
        mp = AugmentedMathematicalProgram()
                           
        num_boats = state_initial.shape[0]
        print num_boats
        
        #initialize time variables

        if time_cost:
            finish_time = mp.NewContinuousVariables(1, 'finish_time')[0]
        else:
            finish_time = maximum_time
            
        step = finish_time/float(N)
        
        if step_initialization is not None:
            mp.SetInitialGuess(finish_time, step_initialization*N)     
            
        time_array = np.arange(0.0, N+1)*step
        
        
        #initialize state variables
        
        problem_state_initial = self.boat.toProblemState(state_initial)
        problem_state_final  = self.boat.toProblemState(state_final)
                    
        boats_S = mp.states(N+1, problem_state_initial, states_final=problem_state_final)        
        boats_U = mp.inputs(N, self.boat.num_inputs, num_boats=num_boats)
        
        if states_initialization is not None:
            problem_states_initialization = self.boat.toProblemStates(states_initialization)
            for i in range(num_boats):
                mp.SetInitialGuess(boats_S[i], problem_states_initialization[i])
                
        print bold('INITIALIZED %d %s boats') % (num_boats, self.boat.__class__.__name__) +'\nboats_S:%s, boats_U:%s, time_array:%s' % (boats_S.shape, boats_U.shape, time_array.shape)
        print 'Number of decision vars', mp.num_vars()        
        print '%f seconds' % (time.time()-start)


        #Costs
        ######
        
        print bold('ADDING COSTS')
        
        start=time.time()
        
        if time_cost:
            mp.AddQuadraticCost(np.sum(finish_time**2))
        if input_cost:
            mp.AddQuadraticCost(np.sum(boats_U*boats_U))
        if dif_states_cost:
            self.boat.add_dif_distance_costs(mp, boats_S, states_final)
            
        print '%f seconds' % (time.time()-start)

        #Constraints
        ############
        
        print bold('ADDING CONSTRAINTS')
        print self.boat.linear
        
        start=time.time()
                
        if time_cost:
            mp.AddLinearConstraint(finish_time >= minimum_time)
            mp.AddLinearConstraint(finish_time <= maximum_time)

        self.boat.add_collision_constraints(mp, boats_S)

        for b in range(num_boats):
            for k in range(1,N+1):
                u0 = boats_U[b,k-1] #old input
                s0 = boats_S[b,k-1] #old state
                s  = boats_S[b,k]   #new state

                #State transition constraint
                mp.add_equal_constraints(s, s0 + step*self.boat.boat_dynamics(s0, u0),linear=self.boat.linear, slack=slack)             
                for i in range(self.boat.num_inputs):
                    mp.AddLinearConstraint(u0[i] >= -self.boat.max_u)
                    mp.AddLinearConstraint(u0[i] <=  self.boat.max_u)    
                    
        print '%f seconds' % (time.time()-start)
        
                
        #Calculate Solution 
        ###################
        
        start=time.time()

        print bold('PLANNING')
        result = mp.Solve()
        print result
        solve_time = time.time()-start
        print '%f seconds' % (solve_time)
        print

        boats_U = np.array([mp.GetSolution(U) for U in boats_U])
        boats_S = self.boat.toGlobalStates(np.array([mp.GetSolution(S) for S in boats_S]), state_initial)
        if time_cost:
            time_array = np.array([x.Evaluate({finish_time: mp.GetSolution(finish_time)}) for x in time_array])

        return boats_S, boats_U, time_array[-1], mp, result, solve_time


    def compute_spline_trajectory(self, minimum_time, maximum_time, state_initial, state_final, input_position_cost=False, input_angle_cost=False, dif_states_cost=False, states_initialization=None, step_initialization=None, fix_initialization_inds=None, in_hull=None, on_edge=None, N=100, slack=1e-3):
                
        start = time.time()
        
        #Check variable dimensions
        ##########################
        
        assert state_initial.shape==state_final.shape   # Same number of boats, state vars for final and initial states
        
        #Initialize Optimization Variables
        ##################################       
        
        mp = AugmentedMathematicalProgram()
                           
        num_boats = state_initial.shape[0]
        
        #initialize time variables
        
        finish_time = maximum_time
        step = finish_time/float(N)
        
        T = N+1
        time_array = np.arange(0.0, T)*step

        #initialize state variables

        problem_state_initial = self.boat.toProblemState(state_initial)
        problem_state_final   = self.boat.toProblemState(state_final)
        
        boats_S = mp.states(N+1, problem_state_initial, states_final=problem_state_final)        
        boats_U = mp.inputs(N, self.boat.num_inputs, num_boats=num_boats)
        
        if states_initialization is not None:
            problem_states_initialization = self.boat.toProblemStates(states_initialization)
            
            for i in range(num_boats):
                mp.SetInitialGuess(boats_S[i], problem_states_initialization[i])
            if fix_initialization_inds is not None:
                mp.fix_initialization(boats_S, states_initialization, fix_initialization_inds)
        
        opt_hull = False
        
        if in_hull is None:
            opt_hull=True
            in_hull = mp.NewBinaryVariables(T-1, len(self.boat.hull_path), "hull")
            
        if on_edge is None:
            on_edge = mp.NewBinaryVariables(T-2, 1, "edge")            

        print bold('INITIALIZED %d %s boats') % (num_boats, self.boat.__class__.__name__) +'\nboats_S:%s, boats_U:%s, time_array:%s' % (boats_S.shape, boats_U.shape, time_array.shape)
        print 'Number of decision vars', mp.num_vars()        
        print '%f seconds' % (time.time()-start)
        
        ##Calculate Quadratic B-Spline Parameters
        
        M = 0.5 * np.array([[1, 1, 0],[-2, 2, 0],[1, -2, 1]])
        arr = np.array([[0, 0, 2]])        
        Q  = arr.T.dot(arr)
        deriv_mat = M.T.dot(Q.dot(M))

        #Costs
        ######
        
        print bold('ADDING COSTS')
        
        start=time.time()
        
        if input_position_cost:
            for k in range(T):
                P = np.zeros((3,2),boats_S.dtype)
                if k==0:
                    P[0] = P[1] = boats_S[0,0,:2]
                    P[2] = boats_S[0,k+1,:2]
                elif k>N-1:
                    P[0] = boats_S[0,k-1,:2]
                    P[1] = P[2] = boats_S[0,N,:2]
                else:
                    P=boats_S[0,k-1:k+2,:2]
                print P
                mp.AddQuadraticCost(np.sum(np.multiply(deriv_mat.dot(P),P)))
                
        if input_angle_cost:
            mp.AddQuadraticCost(np.sum(boats_U[0,:,2:]**2/1000))
                                
        print '%f seconds' % (time.time()-start)
        
        #Constraints
        ############
        
        print bold('ADDING CONSTRAINTS')        
        start=time.time()

        self.boat.add_collision_constraints(mp, boats_S, in_hull, on_edge, opt_hull=opt_hull)
        
        angle_mod = mp.NewBinaryVariables(N, 2, "angle_mod")
        self.boat.angle_mod = angle_mod
        mp.add_leq_constraints(angle_mod[:,0]+angle_mod[:,1],np.ones(N))
        
        #angular velocity update
        mp.add_equal_constraints(boats_S[0,1:,5],boats_S[0,:-1,5]+.50*boats_U[0,:,2]+.50*boats_U[0,:,3],linear=True)
        
        #angle update
        mp.add_equal_constraints(boats_S[0,1:,2],boats_S[0,:-1,2]+(angle_mod[:,0]-angle_mod[:,1])*360+\
                                 boats_S[0,:-1,5]+.375*boats_U[0,:,2]+.125*boats_U[0,:,3],linear=True)

        print '%f seconds' % (time.time()-start)
                
        #Calculate Solution 
        ###################
        
        start=time.time()

        print bold('PLANNING')
        result = mp.Solve()
        print result
        solve_time = time.time()-start
        print '%f seconds' % (solve_time)
        print
        
        boats_U = np.array([mp.GetSolution(U) for U in boats_U])
        boats_S = self.boat.toGlobalStates(np.array([mp.GetSolution(S) for S in boats_S]), state_initial)
                
        if opt_hull:
            in_hull = mp.GetSolution(in_hull).reshape(in_hull.shape)
            on_edge = mp.GetSolution(on_edge).reshape(on_edge.shape)
        
        return boats_S, boats_U, in_hull, on_edge, mp, result, solve_time

def write_experiment(boat, boats_S, boats_U, label):
    with open('results/path_'+label+'.pickle', 'wb') as f:
        pickle.dump([boat,boats_S,boats_U], f)
    

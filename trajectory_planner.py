from augmented_mp import AugmentedMathematicalProgram
from pydrake.all import MosekSolver
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


    def compute_mip_trajectory(self, S_initial, S_final, S_initialization=None, U_initialization=None, in_hull=None, S_fix_inds=None, N=15, opt_angle=True, opt_position=True):
                
        start = time.time()
        
        #Check variable dimensions
        ##########################
        
        assert S_initial.shape==S_final.shape   # Same number of boats, state vars for final and initial states
        assert S_initial.shape[0]==1
        
        #Initialize Optimization Variables
        ##################################       
        
        mp = AugmentedMathematicalProgram() 
        #mp.SetSolverOption(MosekSolver().solver_type(), 'MSK_DPAR_MIO_TOL_REL_GAP', 2)
        T  = N+1

        #initialize state variables
        problem_S_initial = self.boat.toProblemState(S_initial)
        problem_S_final   = self.boat.toProblemState(S_final)
        
        boats_S = mp.states(N+1, problem_S_initial, states_final=problem_S_final)
        boats_U = mp.inputs(N, self.boat.num_inputs)
        
        if S_initialization is not None:
            problem_S_initialization = self.boat.toProblemStates(S_initialization)

            mp.SetInitialGuess(boats_S[0], problem_S_initialization[0])
            
            if S_fix_inds is None:
                S_fix_inds = []
                
                if not opt_angle:
                    S_fix_inds += [2,5]
                
                if not opt_position:
                    S_fix_inds += [0,1,3,4]
            
            if S_fix_inds is not None:
                mp.fix_initialization(boats_S, problem_S_initialization, S_fix_inds)

        if U_initialization is not None:
            
            mp.SetInitialGuess(boats_U[0], U_initialization[0])
            
            if not opt_angle:
                mp.fix_initialization(boats_U, U_initialization, [2,3])

            if not opt_position:
                mp.fix_initialization(boats_U, U_initialization, [0,1])

                
        # integer variables
        opt_hull = in_hull is None
        if opt_hull:
            in_hull  = mp.NewBinaryVariables(T-1, len(self.boat.hull_path), "hull")
            
            mp.add_equal_constraints(np.sum(in_hull, axis=1), np.ones(in_hull.shape[0]))
            
        angle_mod = np.zeros((N,2)) 
        #mp.add_leq_constraints(angle_mod[:,0]+angle_mod[:,1],np.ones(N))

        print bold('INITIALIZED %s ') % (self.boat.__class__.__name__) + \
              '\nboats_S:%s, boats_U:%s' % (boats_S.shape, boats_U.shape)    
        print 'Number of decision vars', mp.num_vars()        
        print '%f seconds' % (time.time()-start)


        #Costs
        ######
        
        print bold('ADDING COSTS')  
        start=time.time()
        
        if opt_position:
            self.boat.add_input_position_cost(boats_S, boats_U, mp)

        if opt_angle:
            self.boat.add_input_angle_cost(boats_S, boats_U, mp)

        print 'Number of costs', len(mp.quadratic_costs())                                  
        print '%f seconds' % (time.time()-start)

        
        #Constraints
        ############
        
        print bold('ADDING CONSTRAINTS')        
        start=time.time()
        
        if opt_position:
            self.boat.add_position_collision_constraints(boats_S, in_hull, opt_hull, mp)

        if opt_angle:
            self.boat.add_angle_collision_constraints(boats_S, in_hull, opt_hull, mp)
        
        print S_fix_inds
        self.boat.add_transition_constraints(boats_S, boats_U, angle_mod, mp, S_fix_inds)

        if opt_hull:
            self.boat.add_integer_constraints(in_hull, mp)
            
        print 'Number of constraints', len(mp.linear_constraints())                      
        print '%f seconds' % (time.time()-start)

        
        #Calculate Solution 
        ###################
        
        start=time.time()
        print bold('PLANNING')
        
        result = mp.Solve()
        solve_time = time.time()-start
        
        print result
        print "Solver: %s" % mp.GetSolverId().name()
        print '%f seconds' % solve_time
        print
        
        boats_U = np.array([mp.GetSolution(U) for U in boats_U])
        boats_S = self.boat.toGlobalStates(np.array([mp.GetSolution(S) for S in boats_S]), S_initial)
                
        if opt_hull:
            in_hull = np.array(mp.GetSolution(in_hull).reshape(in_hull.shape), dtype=int)
        
        return boats_S, boats_U, in_hull, mp, result, solve_time

def write_experiment(boat, boats_S, boats_U, label):
    with open('results/path_'+label+'.pickle', 'wb') as f:
        pickle.dump([boat,boats_S,boats_U], f)

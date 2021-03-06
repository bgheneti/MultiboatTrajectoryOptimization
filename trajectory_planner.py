from augmented_mp import AugmentedMathematicalProgram
import numpy as np
import time

def bold(x):
    return '\033[1m'+ x + '\033[0m'


class BoatConfigurationPlanning(object):

    def __init__(self, boat):
        self.boat = boat 
                
    def compute_trajectory(self, minimum_time, maximum_time, state_initial, state_final, time_cost=True, input_cost=False, dif_states_cost=False, states_initialization=None, step_initialization=None):
        
        start = time.time()
        
        #Check variable dimensions
        ##########################
        
        assert state_initial.shape==state_final.shape   # Same number of boats, state vars for final and initial states
        
        
        #Initialize Optimization Variables
        ##################################       
        
        mp = AugmentedMathematicalProgram()
                           
        num_boats = state_initial.shape[0]               
        N = 100
        
        
        #initialize time variables

        if time_cost:
            finish_time = mp.NewContinuousVariables(1, 'finish_time')[0]
        else:
            finish_time = maximum_time
            
        step = finish_time/(N)
        
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
            mp.AddQuadraticCost(100*u0.dot(u0))
        if dif_states_cost:
            self.boat.add_dif_distance_costs(mp, boats_S, states_final)
            
        print '%f seconds' % (time.time()-start)

            
        #Constraints
        ############
        
        print bold('ADDING CONSTRAINTS')
        
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
                mp.add_equal_constraints(s, s0 + step*self.boat.boat_dynamics(s0, u0))             
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
        finish_time_val = mp.GetSolution(finish_time)
        time_array = np.array([x.Evaluate({finish_time:finish_time_val}) for x in time_array])
        
        return boats_S, boats_U, time_array[-1], mp, result, solve_time
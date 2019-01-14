from pydrake.all import MathematicalProgram, SolverType
from pydrake.symbolic import sin, cos
import numpy as np
from collections import Iterable

class AugmentedMathematicalProgram(MathematicalProgram):
    
    #generate state and input matrices
    ##################################
    
    def set_max_iters(max_solver_iterations):
        self.SetSolverOption(SolverType.kSnopt, 'MajorIterationsLimit', max_solver_iterations)
        
    def var_stack(self, N, num_states, name):
        return self.NewContinuousVariables(N, num_states, name)
        
    def states(self, N, states_initial, states_final=None):
        num_boats = states_initial.shape[0]
            
        #intialize states for a boat
        def boat_state(b):
            S = self.var_stack(N, len(states_initial[b]), 'b_%d-s' % b)
            self.add_equal_constraints(S[0], states_initial[b])
            if states_final is not None:
                self.add_equal_constraints(S[-1], states_final[b], linear=True)
            return S
                
        boats_S = np.stack((boat_state(b) for b in range(num_boats)))

        return boats_S
    
    def fix_initialization(self, S, initialization, state_inds):
        for i,s in enumerate(S):
            for j,s_j in enumerate(s):
                self.add_equal_constraints(s_j[state_inds], initialization[i,j,state_inds], linear=True)
        
    def inputs(self, N, num_inputs, num_boats=1):
        boats_U = np.stack(self.var_stack(N, num_inputs, 'b_%d-u' % b) for b in range(num_boats))

        return boats_U
    
    #constraint functions
    #####################
    
    def pairwise_constraints(self, state, val, f):
        try:
            return [f(x, y) for x,y in zip(state, val)] if isinstance(state, Iterable) else f(state, val)
        except:
            print state, val, f
            raise
    
    def add_equal_constraints(self, state, val, linear=False, slack=0.):
        f_nonlinear = lambda x,y: self.AddConstraint(x == y)
        f_linear    = lambda x,y: self.AddLinearConstraint(x == y)
        f_linear    = lambda x,y: [self.AddLinearConstraint(x >= y-slack), self.AddLinearConstraint(x <= y+slack)]
            
        return self.pairwise_constraints(state, val, f_linear if linear else f_nonlinear)
    
    def add_leq_constraints(self, state, val, linear=False):
        f_nonlinear = lambda x,y: self.AddConstraint(x <= y)
        f_linear    = lambda x,y: self.AddLinearConstraint(x <= y)
            
        return self.pairwise_constraints(state, val, f_linear if linear else f_nonlinear)
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
                self.add_equal_constraints(S[-1], states_final[b])
            return S
                
        boats_S = np.stack((boat_state(b) for b in range(num_boats)))

        return boats_S
        
    def inputs(self, N, num_inputs, num_boats=1):
        boats_U = np.stack(self.var_stack(N, num_inputs, 'b_%d-u' % b) for b in range(num_boats))

        return boats_U
    
    #constraint functions
    #####################
    
    def add_equal_constraints(self, state, val, linear=False):
        slack = 0.0001
        def add_equal_constraint(x, y):
            self.AddConstraint(x == y)
            
        def add_equal_linear_constraint(x, y):
            self.AddLinearConstraint(y-slack == x)

        if isinstance(state, Iterable):
            for x,y in zip(state, val):
                add_equal_linear_constraint(x, y) if linear else add_equal_constraint(x, y)
        else:
            add_equal_constraint(state, val)
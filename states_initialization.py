import numpy as np

from scipy.optimize import linear_sum_assignment
from icp.icp import best_fit_transform
from boat_models import Boat

#Take Start and Goals x,y and provide matching goal index for each start
def hungarian_assignment(starts, goals):
    n_boats = starts.shape[0]
    S = np.zeros(starts.shape)
    G = np.zeros(goals.shape)
    starts_tile = starts.reshape((starts.shape[0],1,starts.shape[1]))
    starts_tile = np.tile(starts_tile, (1, n_boats, 1))
    goals_tile = np.tile(goals, (n_boats, 1, 1))
    
    D = np.sum((starts_tile - goals_tile)**50, axis=2)
    
    return linear_sum_assignment(D)[1]

#set velocity of state as difference between next and present state
def set_velocity(states, dt=0.2):
    steps = states.shape[1]
    num_states = states.shape[2]
    for i in range(steps-1):
        states[:,i,num_states/2:] = (states[:,i+1,:num_states/2]-states[:,i,:num_states/2])/dt

#interpolate shape x,y by transforming by interpolation of rotation and offset
def interpolate_shape_transform(end_shape, theta, offset, N, all_offset, num_states):
    states = np.zeros((end_shape.shape[0], N+1, end_shape.shape[1]))
    step_theta = 0
    step_t = np.zeros(offset.shape)
    end_shape_normalized = end_shape.T-all_offset.T
    
    for i in range(1, N+2):
        step_R = np.array([[np.cos(step_theta),-np.sin(step_theta)],[np.sin(step_theta),np.cos(step_theta)]])
        states[:,-i,:2] = (step_R.dot(end_shape_normalized)+step_t.T+all_offset.T).T
        step_t += offset/N
        step_theta += theta/N   
    return states

#use icp from xN to x0 to initialize a path from closest point fit to x0 to xN
def icp_shape_initialization(x0, xN, N, dt):
    num_boats = x0.shape[0]
    num_states = x0.shape[1]
    
    end_shape_mean = np.mean(xN[:,:2], axis=0).reshape(-1,2)
    T,R,t = best_fit_transform(xN[:,:2]-end_shape_mean, x0[:,:2]-end_shape_mean)
    theta = -np.arccos(R[0,0])
    offset = t.reshape((-1, len(t)))
    
    start_shape = T.dot( np.vstack(((xN[:,:2]-end_shape_mean).T, np.ones(num_boats))) ).T[:,:2]
    print start_shape
    start_shape += end_shape_mean
    start_state = np.zeros((num_boats, num_states))
    start_state[:,:2] = start_shape

    return interpolate_shape_transform(xN[:,:2], theta, offset, N, end_shape_mean, num_states)

#expand x by centroid of points in x                 
def expansion(x, expansion_factor=1):
    x_centroid = np.expand_dims(np.mean(x, axis=0), axis=0)
    dif = x - x_centroid
    return x_centroid+dif*expansion_factor

#interpolate between last state in first states array and first state in second states array. return all states concatenated
def interpolate(start_S, end_S, N=0):
    start_len = start_S.shape[1]
    end_len = end_S.shape[1]
                           
    boats_S = np.zeros((start_S.shape[0], start_len + N + end_len, start_S.shape[-1]))
    boats_S[:,:start_len,:] = start_S
    boats_S[:,-end_len:,:] = end_S

    last_start_ind = start_len - 1
    
    for i in range(1, N+1):
        boats_S[:,last_start_ind+i,:] = (i*end_S[:,0,:] + (N-i)*start_S[:,-1,:])/float(N+1)
    return boats_S

#initialization states for mathematical program
def states_initialization(x0, xN, N, dt, min_assignment=True, shapemove_N=None, expansion_N=None, expansion_factor=None):
    num_boats = x0.shape[0]
    num_states = x0.shape[1]

    boats_S = np.zeros((num_boats, N+1, num_states))
    boat_assignments = np.arange(num_boats)
                       
    move_N = N-3

    start_S = np.tile(x0.reshape(num_boats, 1, num_states), (1,2,1))
    end_S = np.tile(xN.reshape(num_boats, 1, num_states), (1,2,1))
        
    if expansion_N is not None:
        move_N -= expansion_N     
        assert expansion_N % 2 == 0
        assert expansion_factor is not None
        
        start_S = interpolate(start_S, np.expand_dims(expansion(x0, expansion_factor), 1), expansion_N/2-1)
        end_S = interpolate(np.expand_dims(expansion(xN, expansion_factor), 1), end_S, expansion_N/2-1)
                                              
    if shapemove_N is not None:
        move_N -= shapemove_N
        shape_S = np.zeros((num_boats, shapemove_N, num_states))
        shape_S[:,:,:2] = icp_shape_initialization(start_S[:,-1], end_S[:,0], shapemove_N, dt)[:,:-1,:]
        end_S = interpolate(shape_S, end_S)
            
    if min_assignment:
        boat_assignments = hungarian_assignment(start_S[:,-1,:2], end_S[:,0,:2])
        end_S =  end_S[boat_assignments]

    assert move_N >= 0
    boats_S = interpolate(start_S, end_S, move_N)
    set_velocity(boats_S, dt)
    
    return (boats_S, boat_assignments)
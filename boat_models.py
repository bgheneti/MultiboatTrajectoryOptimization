import numpy as np
from pydrake.symbolic import sin, cos
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt


class Boat():   
    G  = 9.8  # gravitational constant    
    num_states = 6
    num_inputs = 4
    
    d11 = 6
    d22 = 8
    d33 = 0.6
    m11 = 12
    m22 = 16
    m33 = 1.5
    width = 0.9
    height = 0.45
    Q0 = 50
    Q1 = 50
    Q2 = 5
    max_u = 4
    linear=False
    
    def __init__(self, split=False, min_interboat_clearance=0):
        self.split = split
        self.min_interboat_clearance = min_interboat_clearance
        self.min_interboat_distance = ((self.height)**2+(self.width)**2)**0.5 + self.min_interboat_clearance
        self.min_interboat_distance_squared = self.min_interboat_distance**2

        self.split_min_interboat_distance = ((self.height)**2+(self.width/2)**2)**0.5 + self.min_interboat_clearance
        self.split_min_interboat_distance_squared = self.split_min_interboat_distance**2
        
    def toProblemState(self, boats_s):
        return boats_s
    
    def toProblemStates(self, boats_S):
        return boats_S
    
    def toGlobalStates(self, boats_S, state_initial=None):
        return boats_S
    
    @classmethod
    def boat_dynamics(cls, s, u):
        '''
        Calculates the dynamics, i.e.:
           \dot{state} = f(state,u)

        for the rocket + two planets system.

        :param state: numpy array, length 4, comprising state of system:
            [x, y, \dot{x}, \dot{y}]
        :param u: numpy array, length 2, comprising control input for system:
            [\ddot{x}_u, \ddot{y}_u]   
            Note that this is only the added acceleration, note the total acceleration.

        :return: numpy array, length 4, comprising the time derivative of the system state:
            [\dot{x}, \dot{y}, \ddot{x}, \ddot{y}]
        '''

        derivs = np.zeros_like(s)
        
        derivs[0] = cos(s[2]) * s[3] - sin(s[2]) * s[4];
        derivs[1] = sin(s[2]) * s[3] + cos(s[2]) * s[4];
        derivs[2] = s[5];
        derivs[3] = -cls.d11 / cls.m11 * s[3] + u[0] / cls.m11 + u[1] / cls.m11;
        derivs[4] = -cls.d22 / cls.m22 * s[4] + u[2] / cls.m22 + u[3] / cls.m22;
        derivs[5] = -cls.d33 / cls.m33 * s[5] + cls.width / (2 * cls.m33) * u[0] - cls.width / (2 * cls.m33) * u[1] + cls.height / (2 * cls.m33) * u[2] - cls.height / (2 * cls.m33) * u[3];
        return derivs    
    
    @classmethod
    def passive_boat_dynamics(cls, state):
        '''
        Caculates the dynamics with no control input, see documentation for boat_dynamics
        '''
        u = np.zeros(2)
        return cls.boat_dynamics(state, u)

    @classmethod
    def two_norm(cls, x):
        '''
        Euclidean norm but with a small slack variable to make it nonzero.
        This helps the nonlinear solver not end up in a position where
        in the dynamics it is dividing by zero.

        :param x: numpy array of any length (we only need it for length 2)
        :return: numpy.float64
        '''
        slack = .001
        return np.sqrt(((x)**2).sum() + slack)

    @classmethod
    def simulate_states_over_time(cls, state_initial, time_array, input_trajectory):
        '''
        Given an initial state, simulates the state of the system.

        This uses simple Euler integration.  The purpose here of not
        using fancier integration is to provide what will be useful reference for
        a simple direct transcription trajectory optimization implementation.

        The first time of the time_array __is__ the time of the state_initial.

        :param state_initial: numpy array of length 4, see boat_dynamics for documentation
        :param time_array: numpy array of length N+1 (0, ..., N) whose elements are samples in time, i.e.:
            [ t_0,
              ...
              t_N ] 
            Note the times do not have to be evenly spaced
        :param input_trajectory: numpy 2d array of N rows (0, ..., N-1), and 2 columns, corresponding to
            the control inputs at each time, except the last time, i.e.:
            [ [u_0, u_1],
              ...
              [u_{N-1}, u_{N-1}] ]

        :return: numpy 2d array where the rows are samples in time corresponding
            to the time_array, and each row is the state at that time, i.e.:
            [ [x_0, y_0, \dot{x}_0, \dot{y}_0],
              ...
              [x_N, y_N, \dot{x}_N, \dot{y}_N] ]
        '''
        states_over_time = np.asarray([state_initial])
        for i in range(1,len(time_array)):
            time_step = time_array[i] - time_array[i-1]
            state_next = states_over_time[-1,:] + time_step*cls.boat_dynamics(states_over_time[-1,:], input_trajectory[i-1,:])
            states_over_time = np.vstack((states_over_time, state_next))
        return states_over_time

    @classmethod
    def simulate_states_over_time_passive(cls, state_initial, time_array):
        '''
        Given an initial state, simulates the state of the system passively

        '''
        input_trajectory = np.zeros((len(time_array)-1,2))
        return cls.simulate_states_over_time(state_initial, time_array, input_trajectory)

    @classmethod
    def plot_trajectory(cls, trajectory, plot=None, color='black', linestyle='-'):
        '''
        Given a trajectory, plots this trajectory over time.

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        '''
        input_trajectory = np.zeros((trajectory.shape[0],2))
        cls.plot_trajectory_with_inputs(trajectory, input_trajectory, plot=plot, color=color, linestyle=linestyle)

    @classmethod
    def plot_trajectory_with_inputs(cls, trajectory, input_trajectory, plot=None, color='black', linestyle='-'):
        '''
        Given a trajectory and an input_trajectory, plots this trajectory and control inputs over time.

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        :param: input_trajectory: the input to simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the input_trajectory
        '''

        rocket_position_x = trajectory[:,0]
        rocket_position_y = trajectory[:,1]

        #check if fig, axes already exist                   
        if plot is None:
            fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(8,6))
        else:
            fig, axes = plot
                           
        axes.plot(rocket_position_x, rocket_position_y, linewidth='2', color=color, zorder=0, linestyle=linestyle)
        axes.axis('equal')

        ## if we have an input trajectory, plot it
        if len(input_trajectory.nonzero()[0]):
            # the quiver plot works best with not too many arrows
            max_desired_arrows = 40
            num_time_steps = input_trajectory.shape[0]

            if num_time_steps < max_desired_arrows:
                downsample_rate = 1 
            else: 
                downsample_rate = num_time_steps / max_desired_arrows

            rocket_position_x = rocket_position_x[:-1] # don't need the last state, no control input for it
            rocket_position_y = rocket_position_y[:-1]
            rocket_booster_x = input_trajectory[::downsample_rate,0]
            rocket_booster_y = input_trajectory[::downsample_rate,1]
            Q = plt.quiver(rocket_position_x[::downsample_rate], rocket_position_y[::downsample_rate], \
                rocket_booster_x, rocket_booster_y, units='width', color="red")
        
        if plot is None:
            plt.show()
            
    def plot_configurations(self, boats_S, stride=10):
        plot = plt.subplots(nrows=1,ncols=1, figsize=(12,8))
        pick = range(0,boats_S.shape[1],stride)
        num_display = len(pick)
        alphas = np.repeat(np.arange(0,num_display)/float(num_display), len(boats_S))*0.5
        self.plot_configuration(np.vstack([boats_S[:,i,:] for i in pick]), border=0, region_fill=False, boat_color='black', alphas=alphas, plot=plot)
        self.plot_configuration(boats_S[:,-1,:],region_fill=False, border=0, plot=plot, boat_color='black')
        plt.plot()
        
    def plot_x0xN(self, boats_S, alpha0=[0.5], alphaN=[1], plot=None, boat_color0='w', boat_colorN='darkorange', region_color0='0.5', region_colorN='0.2'):
        if plot is None:
            fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(12,8))
        else:
            fig, axs = plot
        self.plot_configuration(boats_S[:,0], alphas=alpha0, plot=(fig,axs), region_color=region_color0, boat_color=boat_color0)
        self.plot_configuration(boats_S[:,-1], alphas=alphaN, plot=(fig,axs), region_color=region_colorN, boat_color=boat_colorN)
        if plot is None:
            plt.plot()
        return fig, axs

    def plot_configuration(self, boats_s, alphas=None, show_regions=True, plot=None, region_color='0.2', boat_color='darkorange', border=None, region_fill=True):
        if plot is None:
            fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(24,16))        
            split = self.split
        else:
            fig, axs = plot
        axs.set_aspect('equal')
        if alphas is None:
            alphas = np.ones(len(boats_s))
        elif len(alphas)==1:
            alphas = alphas[0]*np.ones(len(boats_s))            
                
        def patch_pose(pose):
            x,y,theta = pose
            return (
                    x - self.width/2*np.cos(theta) + self.height/2*np.sin(theta),
                    y - self.width/2*np.sin(theta) - self.height/2*np.cos(theta),
                    theta*180/np.pi
                   )
        def patches_interboat_split(pose):
            x,y,theta = pose
            return  [[x - self.height/2*np.cos(theta), y - self.height/2*np.sin(theta)],
                     [x + self.height/2*np.cos(theta), y + self.height/2*np.sin(theta)]
                    ]
                 
        patches = []
        regions = []
        for pose,alpha in zip(boats_s,alphas):
            if show_regions:
                if self.split:         
                    regions.append([Circle(pose[:3], self.split_min_interboat_distance/2, color=region_color, alpha=alpha),
                                    Circle(pose[:3], self.split_min_interboat_distance/2, color=region_color, alpha=alpha)
                                   ])
                    axs.add_patch(regions[-1][0], alpha=alpha) 
                    axs.add_patch(regions[-1][1], alpha=alpha) 
                else:
                    regions.append(Circle(pose[:3], self.min_interboat_distance/2, color=region_color, alpha=alpha, linewidth=border, linestyle='solid', edgecolor='0', fill=region_fill))
                    axs.add_patch(regions[-1])

            pose = patch_pose(pose[:3])
            patches.append(
                Rectangle(
                    pose[:2],
                    self.width,
                    self.height,
                    pose[2],
                    alpha=alpha,
                    color=boat_color
                )
            )
            axs.add_patch(patches[-1])
        if plot is None:
            plt.plot()
        return patches, regions

    @classmethod                       
    def plot_trajectories(cls, trajectories, input_trajectories=None, plot=None, color='black', linestyle='-'):
        if plot is None:
            fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(12,8))
        else:
            fig, axs = plot
            
        plt.gca().set_prop_cycle(None)
            
        if input_trajectories is None:
            for trajectory in trajectories:
                Boat.plot_trajectory(trajectory, plot=(fig, axs), color=color, linestyle=linestyle)
        else:
            for trajectory, input_trajectory in zip(trajectories, input_trajectories):
                Boat.plot_trajectory_with_inputs(trajectory, input_trajectory, plot=(fig, axs), color=color, linestyle=linestyle)
        if plot is None:
            plt.plot()
        return fig, axs

    def plot_animation(self, boats_S, input_trajectories=None, show_regions=True):
        
        split = self.split
                
        def patch_pose(pose):
            x,y,theta = pose
            return (
                    x - self.width/2*np.cos(theta) + self.height/2*np.sin(theta),
                    y - self.width/2*np.sin(theta) - self.height/2*np.cos(theta),
                    theta*180/np.pi
                   )
        def patches_interboat_split(pose):
            x,y,theta = pose
            return  [[x - self.height/2*np.cos(theta), y - self.height/2*np.sin(theta)],
                     [x + self.height/2*np.cos(theta), y + self.height/2*np.sin(theta)]
                    ]
        
        def animate(frame):
            for p in range(len(patches)):
                patch = patches[p]
                pose = boats_S[p][frame][:3]
                px,py,ptheta = patch_pose(pose)
                patch.set_x(px)
                patch.set_y(py)
                patch._angle = ptheta
                
                if show_regions:
                    region = regions[p]
                    if split:
                        region[0].center, region[1].center = patches_interboat_split(pose)
                    else:
                        region.center = boats_S[p][frame][:3]
            
            if show_regions:
                if split:
                    return patches+[r for r in region for region in regions]
                return patches+regions
            
            return patches
                
        fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(12,8))
        
        axs.axis('off')
        
        self.plot_x0xN(boats_S, [0.2], [0.2], plot=(fig, axs), boat_colorN='w', region_colorN='darkorange')
        
        axs.set_aspect('equal')
        patches, regions = self.plot_configuration(boats_S[:,0], plot=(fig,axs), boat_color='darkorange', border=1, region_color='0.2')
        plt.xlim(np.min(boats_S[:,:,0])-self.width, np.max(boats_S[:,:,0])+self.width)
        plt.ylim(np.min(boats_S[:,:,1])-self.width, np.max(boats_S[:,:,1])+self.width)
        ani = animation.FuncAnimation(fig, animate, frames=range(boats_S.shape[1]), blit=True, interval=20)
        return ani   
    
    def add_collision_constraints(self, mp, boats_S):
        def add_collision_constraint(b1, b2):
            mp.AddConstraint(np.sum((b1[:2]-b2[:2])**2) >= self.min_interboat_distance_squared)

        def add_split_collision_constraint(b1, b2):
            #pairwise collision constraint on each half of boat
            for b1d in [-.25,.25]:
                for b2d in [-.25,.25]:
                    b1x = b1[0]+self.width*b1d
                    b1y = b1[1]+self.width*b1d
                    b2x = b2[0]+self.width*b2d
                    b2y = b2[1]+self.width*b2d
                    mp.AddConstraint((b1x-b2x)**2+(b1y-b2y)**2 >= self.split_min_interboat_distance_squared)
        
        for i in range(len(boats_S)):
            for j in range(i+1, len(boats_S)):
                for t in range(len(boats_S[0])):
                    if self.split:
                        add_split_collision_constraint(boats_S[i][t], boats_S[j][t])
                    else:
                        add_collision_constraint(boats_S[i][t], boats_S[j][t])
                        
    def add_dif_distance_costs(self, mp, boats_S, final_states):
        for i in range(len(boats_S)):
            for j in range(i+1, len(boats_S)):
                final_state_dif = final_states[i]-final_states[j] #np.expand_dims(final_states[i]-final_states[j], axis=0)
                for t in range(len(boats_S[0])):
                    mp.AddQuadraticCost(np.sum((boats_S[i,t]-boats_S[j,t]-final_state_dif)**2))
                        
class ThreeInputBoat(Boat):
    num_inputs = 3
    max_u = 6
    
    @classmethod
    def boat_dynamics(cls, s, u):
        
        derivs = np.zeros_like(s)
        
        derivs[0] = s[3];
        derivs[1] = s[4];
        derivs[2] = s[5];
        derivs[3] = -cls.d11 / cls.m11 * s[3] + u[0] / cls.m11;
        derivs[4] = -cls.d22 / cls.m22 * s[4] + u[1] / cls.m22;
        derivs[5] = -cls.d33 / cls.m33 * s[5] + cls.width / (2 * cls.m33) * u[2];
        return derivs

class TwoInputBoat(Boat):
    num_inputs = 2
    num_states = 4
    max_u = 6
    state_selector = np.array([0,1,3,4])
    
    def toProblemState(self, boats_s):
        assert boats_s.shape[1] == Boat.num_states
        return boats_s[:,self.state_selector]
    
    def toProblemStates(self, boats_S):
        assert boats_S.shape[-1] == Boat.num_states
        return boats_S[:,:,self.state_selector]
    
    def toGlobalStates(self, boats_S, state_initial=None):
        assert boats_S.shape[-1] == self.num_states
        shape = np.array(boats_S.shape)
        shape[-1] = Boat.num_states
        boats_S_global = np.zeros(shape)
        
        if state_initial is not None:
            boats_S_global[:,:] = np.expand_dims(state_initial, 1)
            
        boats_S_global[:,:,self.state_selector] = boats_S
        
        return boats_S_global
    
    @classmethod
    def boat_dynamics(cls, s, u):

        derivs = np.zeros_like(s)
        
        derivs[0] = s[2];
        derivs[1] = s[3];
        derivs[2] = -cls.d11 / cls.m11 * s[2] + u[0] / cls.m11;
        derivs[3] = -cls.d22 / cls.m22 * s[3] + u[1] / cls.m22;
        return derivs
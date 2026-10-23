from config import *
import math
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def compute_velocity_and_orientation_from_positions(x1, x2, random_vel_range = [1.5, 1.51]):
    temp = x2 - x1
    ang = np.arctan2(temp[1], temp[0])
    if ang < 0:
        ang += 2. * math.pi
    vel = np.random.uniform(random_vel_range[0], random_vel_range[1])
    return vel, ang

def random_within_circle(radius, low = 0.):
    theta = np.random.uniform(0, 2 * math.pi)
    distance = np.random.uniform(low, radius)
    return np.array([distance * math.cos(theta), distance * math.sin(theta)])

def compute_min_distance2pedestrian(state, pedestrian_positions):
    min_distance2pedestrian = min(np.linalg.norm(pedestrian_positions - state[:2], axis = 1))
    return min_distance2pedestrian

def compute_distance2goal(state, env):
    return np.linalg.norm(state[:2] - env.vehicle_goal)

def normalize_state_sequence(local_xs, normalizer, update = True):
    flattened_local_xs = copy.deepcopy(local_xs.reshape([-1, local_xs.shape[-1]]))
    if update:
        normalizer.experience(flattened_local_xs)
    normalized = normalizer(flattened_local_xs)
    return flattened_local_xs.reshape(local_xs.shape)

def get_all_ac_grid(grid_side_size = 50,
                    maxVelAcc = 0.5, maxAngAcc = 0.5 * math.pi):
    vel_acs = np.arange(-maxVelAcc, maxVelAcc, (maxVelAcc * 2.) / grid_side_size)
    ang_acs = np.arange(-maxAngAcc, maxAngAcc, (maxAngAcc * 2.) / grid_side_size)
    acs = np.array(list(itertools.product(vel_acs, ang_acs)))
    return acs

def get_displacement(state, ac,
                     maxVelAcc = 0.5, maxAngAcc = 0.5 * math.pi,
                     minVel = 1.5, maxVel = 1.51,):
    ac = np.clip(ac, [-maxVelAcc, -maxAngAcc], [maxVelAcc, maxAngAcc])

    delta_state = [
        (min(state[2] + ac[0], maxVel)) * np.cos(state[3] + ac[1]),
        (min(state[2] + ac[0], maxVel)) * np.sin(state[3] + ac[1]),
        ac[0],
        ac[1]
    ]
    if state[-1] + ac[1] > 2. * math.pi:
        delta_state[-1] -= 2. * math.pi
    elif state[-1] + ac[1] < 0:
        delta_state[-1] += 2. * math.pi
    return np.array(delta_state)

def step_vehicle(ac, state, 
                 minVel = 1.5, maxVel = 1.51,):
    delta_state = get_displacement(state, ac)
    temp_state = copy.deepcopy(state + np.array(delta_state))
    if temp_state[2] > maxVel:
        temp_state[2] = maxVel 
    elif temp_state[2] < minVel:
        temp_state[2] = minVel
    return temp_state

def predict_next_states_from_actions(acs, state):
    next_states = []
    for ac in acs:
        next_states.append(step_vehicle(ac, state))
    return np.array(next_states)

def normalize_data(safe_local_xs, safe_local_nxs, unsafe_local_xs,
                  safe_vehicle_xs, safe_vehicle_nxs, unsafe_vehicle_xs,
                  local_normalizer, vehicle_normalizer, update = False):
    normalized_safe_local_xs = normalize_state_sequence(safe_local_xs, local_normalizer,
                                              update = update).astype(np.float32)
    normalized_safe_local_nxs = normalize_state_sequence(safe_local_nxs, local_normalizer,
                                                  update = False).astype(np.float32)
    normalized_unsafe_local_xs = normalize_state_sequence(unsafe_local_xs, local_normalizer,
                                                    update = False).astype(np.float32)
    
    if update:
        vehicle_normalizer.experience(safe_vehicle_xs)
        vehicle_normalizer.experience(unsafe_vehicle_xs)
    normalized_safe_vehicle_xs = vehicle_normalizer(safe_vehicle_xs,
                                                update = False).astype(np.float32)
    normalized_safe_vehicle_nxs = vehicle_normalizer(safe_vehicle_nxs,
                                                     update = False).astype(np.float32)
    normalized_unsafe_vehicle_xs = vehicle_normalizer(unsafe_vehicle_xs,
                                                      update = False).astype(np.float32)
    
    data = [normalized_safe_local_xs, normalized_safe_local_nxs, normalized_safe_vehicle_xs, normalized_safe_vehicle_nxs, normalized_unsafe_local_xs, normalized_unsafe_vehicle_xs]
    return data

def load_dynamics_training_data(all_vehicle_positions, all_vehicle_acs, train_ratio = 0.9):
    dynamics_xs, dynamics_ys = [], []
    actions = []
    for traj_vehicle_positions, traj_vehicle_acs in zip(all_vehicle_positions, all_vehicle_acs):
        actions += list(traj_vehicle_acs)
        
        traj_vehicle_positions = np.array(traj_vehicle_positions)
        dynamics_xs += list(np.hstack([traj_vehicle_positions[:-1, -2:], traj_vehicle_acs]))
        dynamics_ys += list(traj_vehicle_positions[1:, :2] - traj_vehicle_positions[:-1, :2]) # Relative displacement

    dynamics_xs = np.array(dynamics_xs)
    dynamics_ys = np.array(dynamics_ys)
    actions = np.array(actions)
    train_size = int(train_ratio * dynamics_xs.shape[0])
    train_xs, train_ys = dynamics_xs[:train_size], dynamics_ys[:train_size]
    val_xs, val_ys = dynamics_xs[train_size:], dynamics_ys[train_size:]
    
    return train_xs, train_ys, val_xs, val_ys

def FC_predict_agent_dynamics(state, ac, dynamics_model):
    base = state[:2]
    x = np.hstack([state[2:], ac]).reshape([1, -1])
    
    predicted_y = dynamics_model.predict(x).reshape([-1]) + base
    temp = np.concatenate([predicted_y, [1.5, state[-1] + ac[-1]]])
    
    if temp[-1] > 2 * math.pi:
        temp[-1] -= 2 * math.pi 
    elif temp[-1] < 0:
        temp[-1] += 2 * math.pi
    return temp

def FC_predict_agent_dynamics_batchActions(state, acs, dynamics_model):
    base = state[:2]
    
    temp = copy.deepcopy(state).reshape([1, -1])
    temp = np.tile(temp, [len(acs), 1])
    x = np.hstack([temp[:, 2:], acs])

    predicted_ys = dynamics_model.predict(x)    
    predicted_ys[:, :2] += base 
    temp = np.hstack([predicted_ys, np.ones([len(predicted_ys)]).reshape([-1, 1]) * 1.51, (state[-1] + acs[:, -1]).reshape([-1, 1])])
    
    temp[:, -1][np.where(temp[:, -1] > 2 * math.pi)] -= 2 * math.pi
    temp[:, -1][np.where(temp[:, -1] < 0)] += 2 * math.pi
    return temp

def FC_predict_obstacle_states(obstacle_positions, # 3 * (number of obstacles * 2)
                               base_positions, # (number of obstacles) * 2
                               dynamics_model): 
    obstacle_positions = np.swapaxes(obstacle_positions, 0, 1)
    for i, base_pos in enumerate(base_positions):
        obstacle_positions[i] -= base_pos
    obstacle_positions = obstacle_positions.reshape([-1, 6])
    pred_obstacle_futures = dynamics_model.predict(obstacle_positions)
    assert pred_obstacle_futures.shape == base_positions.shape 
    return pred_obstacle_futures + base_positions

def find_CBF_boundary(local_xs, local_future_xs,
                      model,
                      local_normalizer, vehicle_normalizer,
                      high = 5., 
                      numSamples = 100,
                      diff = 0.1,
                      tolerance = 1., # -> range: [-tolerance, 0]
                      random_vel_range = [1.5, 1.51]
                      ):
    num_row = int((high * 2) / diff)
    all_states = []
    local_xs = np.expand_dims(local_xs, 0)
    
    # Sampling new agent states
    np.random.seed(None)
    radm = np.random.uniform(-high, high, [numSamples, local_xs.shape[-1]])
    positions = radm + local_future_xs[0]
    vels = np.random.uniform(random_vel_range[0], random_vel_range[1], [numSamples, 1])
    angs = np.random.uniform(0, 2. * math.pi, [numSamples, 1])
    all_states = np.hstack([positions, vels, angs])
    
    tiled_xs = np.tile(local_xs, [all_states.shape[0], 1, 1])
    normalized_local_xs = normalize_state_sequence(tiled_xs, local_normalizer, update = False).astype(np.float32)
    normalized_xs = vehicle_normalizer(all_states)
    pred_vals = model.predict([normalized_local_xs, normalized_xs])#.reshape([num_row, num_row])
    boundary_indices = np.where((pred_vals >= -tolerance) & (pred_vals <= 0))
    return all_states[boundary_indices[0]], pred_vals[boundary_indices]

def get_baseline_ac(state, goal, 
                    obstacle_pos,
                    att_coeff = 0.1, rep_coeff = 20000., Q_star = 4.,
                    use_FC = False, FC_inputs = None, FC_base = None, dynamics_model = None):
    if use_FC:
        assert FC_inputs is not None and FC_inputs.shape[0] == Horizon and FC_base is not None
        assert dynamics_model is not None
        preds_nxs = FC_predict_obstacle_states(FC_inputs, FC_base, dynamics_model)
    else:
        preds_nxs = obstacle_pos#.reshape([1, -1])
#     print (preds_nxs.shape, obstacle_pos.shape)

    random_acs = get_all_ac_grid() 
    vehicle_nxs = predict_next_states_from_actions(random_acs, state)
    distances = [compute_min_distance2pedestrian(nx, preds_nxs) for nx in vehicle_nxs]
    U_att = [0.5 * att_coeff * (np.linalg.norm(nx[:2] - goal)**2) for nx in vehicle_nxs]
    def compute_U_rep(d, Q_star):
        if d <= Q_star:
            return 0.5 * rep_coeff * ((1. / d - 1. / Q_star)**2)
        return 0. 
    U_rep = [compute_U_rep(d, Q_star) for d in distances]
    scores = np.array(U_att) + np.array(U_rep)
    i = np.argmin(scores)
    return random_acs[i], i

def generate_ac_training(model, dynamics_model, 
                         local_xs, local_normalizer, vehicle_normalizer,
                         conservative_bound, 
                         grid_side_size = 50,
                         inverseDynamics = True,
                         printVals = False,
                         state = None, numPedestrians = None, goal = None):
    random_acs = get_all_ac_grid(grid_side_size = grid_side_size)
    if inverseDynamics:        
        vehicle_nxs = FC_predict_agent_dynamics_batchActions(state, random_acs, dynamics_model)
    else:
        vehicle_nxs = predict_next_states_from_actions(random_acs, state = state)
    
    scores = -np.linalg.norm(vehicle_nxs[:, :2] - goal, 1, axis = 1) # goal-driven

    relative_local_xs = np.tile(local_xs,
                                [len(vehicle_nxs), 1, 1])    

    all_vals = - model.predict([normalize_state_sequence(relative_local_xs, local_normalizer, update = False).astype(np.float32), 
                              vehicle_normalizer(vehicle_nxs, update = False)]).reshape([-1])
    
    idx = np.argmax(all_vals)
    return max(all_vals) > conservative_bound, random_acs[idx], idx

def online_inference_NCBF(env, model, dynamics_model,
                          local_normalizer, vehicle_normalizer,
                          traj_pedestrain_positions_list, 
                          greedy_threshold = 1.,
                          conservative_threshold = 0.5, 
                          unsafe_threshold = 0.025, render = True, 
                          grid_side_size = 50,
                          consider_distance = 20,
                          baseline_ac = None,):
    state = copy.deepcopy(env.vehicle.state)
    random_acs = env.vehicle.get_all_ac_grid(grid_side_size = grid_side_size)
    vehicle_nxs = predict_next_states_from_actions(random_acs, state)
    scores = -np.linalg.norm(vehicle_nxs[:, :2] - env.vehicle_goal, 1, axis = 1)


    all_local_xs = []
    all_vehicle_nxs = []
    for i in range(env.numPedestrains):
        local_xs = np.array(traj_pedestrain_positions_list[-(Horizon+1):-1])[:, i]
        base = copy.deepcopy(traj_pedestrain_positions_list[-(Horizon + 2)][i])
        relative_local_xs = (local_xs - base).reshape([1, Horizon, -1])
        relative_vehicle_nxs = copy.deepcopy(state) # - base).reshape([1, -1])
        relative_vehicle_nxs[:2] -= base
        relative_vehicle_nxs = relative_vehicle_nxs.reshape([1, -1])
        
        all_local_xs.append(relative_local_xs)
        all_vehicle_nxs.append(relative_vehicle_nxs)
    all_local_xs = np.vstack(all_local_xs)
    all_vehicle_nxs = np.vstack(all_vehicle_nxs)
    all_current_vals_ = model.predict([normalize_state_sequence(all_local_xs, local_normalizer, update = False).astype(np.float32), 
                                       vehicle_normalizer(all_vehicle_nxs, update = False)])#[0][0]
    all_current_vals = - all_current_vals_.reshape([env.numPedestrains, -1])
    
    invalid_indices = np.where(np.linalg.norm(all_vehicle_nxs, axis = 1) > consider_distance)
    care_indices = np.where(all_current_vals.reshape([-1]) <= 2. * greedy_threshold)[0]
    all_current_vals[invalid_indices] = - float('inf')
        
    if np.min(all_current_vals) > greedy_threshold or len(care_indices) == 0:
        ac_idx = np.argmax(scores)
        ac = random_acs[ac_idx]
    else:
        all_vals = []
        all_local_xs = []
        all_vehicle_nxs = []
        for i in care_indices:
            local_xs = np.array(traj_pedestrain_positions_list[-Horizon:])[:, i]
            base = copy.deepcopy(traj_pedestrain_positions_list[-(Horizon + 1)][i])
            relative_local_xs = (local_xs - base).reshape([1, Horizon, -1])
            relative_local_xs = np.tile(relative_local_xs,
                                        [len(vehicle_nxs), 1, 1])
        
            relative_vehicle_nxs = copy.deepcopy(vehicle_nxs) 
            relative_vehicle_nxs[:, :2] -= base
            all_local_xs.append(relative_local_xs)
            all_vehicle_nxs.append(relative_vehicle_nxs)

        all_local_xs = np.vstack(all_local_xs)
        all_vehicle_nxs = np.vstack(all_vehicle_nxs) 

        all_vals = model.predict([normalize_state_sequence(all_local_xs, local_normalizer, update = False).astype(np.float32), 
                                      vehicle_normalizer(all_vehicle_nxs, update = False)])#[0][0]

#         all_vals = all_vals.reshape([env.numPedestrains, -1])
        all_vals = all_vals.reshape([len(care_indices), -1])

        
        neg_vals = copy.deepcopy(- all_vals)
        neg_vals[np.where(neg_vals < 0.0)] = 0. 
        neg_vals[np.where(neg_vals > conservative_threshold)] = 1.# conservative_threshold
#         neg_vals /= conservative_threshold
        max_vals_incre = np.prod(neg_vals, axis = 0)

        if max(max_vals_incre) <= 0:
            return random_acs[np.argmax(max_vals_incre)]
        found = False 
        for i in np.argsort(scores)[::-1]:
            if max_vals_incre[i] > unsafe_threshold:
                found = True
                break
        ac = random_acs[i]
    
        if (not found):
            return random_acs[np.argmax(max_vals_incre)]
    return ac

def plot_single_traj(traj, color, label, label_interval = 5,
                     xytext = (-5,4), text_fs = 12, dt = 1.):
    plt.plot(traj[:, 0], traj[:, 1], color = color, linewidth = 2,
             label = label)
    for counter, state in enumerate(traj):
        if counter % label_interval == 0:
            etiqueta = "{:.0f}s".format(counter * dt)
            plt.annotate(etiqueta,
                         state[:2],
                        textcoords="offset points",
                         xytext = xytext,
                         ha = "center",
                         fontsize = text_fs)
            plt.scatter(state[0], state[1],
                        facecolors = "white",
                        edgecolors = color,
                        zorder = 10, s = 50)
            if counter != len(traj) - 1:
                arrow_direction = np.array([traj[counter+1, 0] - traj[counter, 0], 
                                            traj[counter+1, 1] - traj[counter, 1]])
                arrow_direction = arrow_direction / np.linalg.norm(arrow_direction) * 2
                plt.arrow(state[0], state[1], arrow_direction[0], arrow_direction[1], 
                          head_width=0.4, zorder=5)
                


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def plot_cbf_single_pedestrian(traj_pedestrain_positions_list,
                               traj_vehicle_positions,
                               time_idx, model, local_normalizer, vehicle_normalizer,
                               visual_size = 10,
                            grid_size = 100,
                            levels = [-0.1, 0.0, 0.1],):
    
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1., 1, 256)
    vals[:, 1] = np.linspace(48 / 255., 1, 256)
    vals[:, 2] = np.linspace(48 / 255., 1, 256)
    color_map = ListedColormap(vals)
    
    base = traj_pedestrain_positions_list[time_idx - Horizon - 1]
    pedestrian_state_history = traj_pedestrain_positions_list[(time_idx - Horizon):time_idx].copy()
    robot_state = traj_vehicle_positions[time_idx]
    
    base = traj_pedestrain_positions_list[time_idx - Horizon - 1]
    pedestrian_state_history = traj_pedestrain_positions_list[(time_idx - Horizon):time_idx].copy()
    robot_state = traj_vehicle_positions[time_idx]

    xs = np.linspace(-visual_size, visual_size, grid_size)
    ys = np.linspace(-visual_size, visual_size, grid_size)
    positions = cartesian([xs, ys])
    positions = np.hstack([positions, 
                           (np.ones(positions.shape[0]) * robot_state[2]).reshape([-1, 1]),
                           (np.ones(positions.shape[0]) * robot_state[3]).reshape([-1, 1]),])
    positions[:, :2] -= base 
    pedestrian_inputs = pedestrian_state_history.copy() - base 
    pedestrian_inputs = np.tile(pedestrian_inputs.reshape([1, *pedestrian_inputs.shape]),
                                [positions.shape[0], 1, 1])

    cbf_values = model.predict([normalize_state_sequence(pedestrian_inputs, local_normalizer, update = False).astype(np.float32), 
                               vehicle_normalizer(positions, update = False)])
    cbf_values = cbf_values.reshape([grid_size, grid_size]).T

    contour = plt.contour(xs, ys, cbf_values, levels = levels, linewidths=0.5, colors='k')
    plt.clabel(contour, inline= True, inline_spacing = -1,fmt = '%1.2f',fontsize=15)
    cntr1 = plt.contourf(xs, ys, cbf_values, cmap=color_map)
    
    plt.xlim(-visual_size, visual_size)
    plt.ylim(-visual_size, visual_size)
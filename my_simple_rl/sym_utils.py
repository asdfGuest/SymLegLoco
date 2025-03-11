import torch as th

'''
- Joint Order
    'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
    'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
    'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'
'''


def _interleave(*args:th.Tensor, dim=-1) :
    shape, dtype, device = args[0].shape, args[0].dtype, args[0].device
    new_shape = list(shape)
    new_shape[dim] *= len(args)
    new_tensor = th.empty(size=new_shape, dtype=dtype, device=device)

    for k in range(len(args)) :
        idx = [slice(None) for _ in range(len(new_shape))]
        idx[dim] = slice(k,None,len(args))
        new_tensor[idx] = args[k]
    
    return new_tensor


def merge_action(act_l:th.Tensor, act_r:th.Tensor) :
    '''
    Args:
        action: (2*n_env, 6)
    Returns:
        (n_envs, 12)
    '''
    act_r = act_r.clone()
    act_r[:,:2] = -act_r[:,:2] # FR hip, RR hip
    act = _interleave(act_l, act_r)
    return act


def decompose_observation(obs:th.Tensor, privileged_obs:th.Tensor) :
    '''
    obs : (n_env, n_history, n_obs)
    privileged_obs : (n_env, n_privileged_obs)
    '''
    if obs.shape[2] != 45 or privileged_obs.shape[1] != 3 :
        raise Exception('Observation shape not match to expectation!')
    
    obs = obs.clone()
    privileged_obs = privileged_obs.clone()
    
    angvel = obs[:,:,0:3]
    gravity = obs[:,:,3:6]
    command = obs[:,:,6:9]
    joint_pos = obs[:,:,9:21]
    joint_vel = obs[:,:,21:33]
    action = obs[:,:,33:45]
    linvel_t = privileged_obs[:,0:3]

    obs_l = th.cat([
        angvel,
        gravity,
        command,
        joint_pos,
        joint_vel,
        action,
    ],dim=-1)
    privileged_obs_l = th.cat([
        linvel_t,
    ],dim=-1)

    angvel[:,:,0] = -angvel[:,:,0] # roll
    angvel[:,:,2] = -angvel[:,:,2] # yaw
    gravity[:,:,1] = -gravity[:,:,1] # y
    command[:,:,1:3] = -command[:,:,1:3] # y, yaw
    for joint_tensor in [joint_pos, joint_vel, action] :
        joint_tensor[:,:,0:4] = -joint_tensor[:,:,0:4] # inverse hip
        joint_tensor[:,:,:] = _interleave(joint_tensor[:,:,1::2], joint_tensor[:,:,0::2]) # left/right
    linvel_t[:,1] = -linvel_t[:,1] # y

    obs_r = th.cat([
        angvel,
        gravity,
        command,
        joint_pos,
        joint_vel,
        action,
    ],dim=-1)
    privileged_obs_r = th.cat([
        linvel_t,
    ],dim=-1)

    return (obs_l, obs_r), (privileged_obs_l, privileged_obs_r)

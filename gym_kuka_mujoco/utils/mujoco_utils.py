import mujoco_py

def kuka_subtree_mass(model):
    body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
    body_ids = [model.body_name2id(n) for n in body_names]
    return model.body_subtreemass[body_ids]

def get_qpos_indices(model, joint_names):
    indices = []
    for name in joint_names:
        idx = model.get_joint_qpos_addr(name)
        if isinstance(idx, tuple):
            indices.extend(range(*idx))
        else:
            indices.append(idx)
    return indices

def get_qvel_indices(model, joint_names):
    indices = []
    for name in joint_names:
        idx = model.get_joint_qvel_addr(name)
        if isinstance(idx, tuple):
            indices.extend(range(*idx))
        else:
            indices.append(idx)
    return indices

def get_actuator_indices(model, actuator_names):
    return [model.actuator_name2id(name) for name in actuator_names]

def get_joint_indices(model, joint_names):
    return [model.joint_name2id(name) for name in joint_names]
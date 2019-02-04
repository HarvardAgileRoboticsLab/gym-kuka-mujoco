import mujoco_py

def kuka_subtree_mass(model):
    body_names = ['kuka_link_{}'.format(i + 1) for i in range(7)]
    body_ids = [model.body_name2id(n) for n in body_names]
    return model.body_subtreemass[body_ids]
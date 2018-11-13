from gym_kuka_mujoco.utils.gen_hole import *
import time
import mujoco_py

def render_mujoco(model_path):
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    while True:
        viewer.render()
        time.sleep(.01)

def test_make_box():
    # Define a test box.
    box_data = {
        'type':'box',
        'pos':(0,0,0),
        'euler':(0,0,0),
        'size':(1, 1, 1)
    }

    box_xml = assemble_mujoco_xml([box_data], attach_worldbody=True)
    model_path = "tmp/box.xml"
    write_xml(model_path, box_xml)
    render_mujoco(model_path)

def test_make_hole():
    # Define a hole
    hole_geoms = gen_hole(.01, .03, .01, 16, radians=False)
    hole_xml = assemble_mujoco_xml(hole_geoms, attach_worldbody=True)
    model_path = 'tmp/hole.xml'
    write_xml(model_path, hole_xml)

    # Render in MuJoCo.
    render_mujoco(model_path)

if __name__=='__main__':
    # test_make_box()
    test_make_hole()
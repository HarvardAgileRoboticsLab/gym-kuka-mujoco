import os
import fnmatch
import re
from string import Template

def get_hole_files():
    hole_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'envs', 'assets', 'hole')
    return sorted([file for file in os.listdir(hole_dir) if fnmatch.fnmatch(file, '*id=*.xml')])

def get_experiment_files():
    hole_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'envs', 'assets')
    return sorted([file for file in os.listdir(hole_dir) if fnmatch.fnmatch(file, '*id=*.xml')])

def gen_experiment_files(hole_files):
    # Get the template file as a string.
    experiment_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'envs', 'assets')
    template_filename = os.path.join(experiment_dir, 'full_peg_insertion_experiment_moving_template.xml')
    with open(template_filename) as template_file:
        template_data = template_file.read()
    
    gravity_options = [('enable', ''), ('disable', '_no_gravity')]

    # Convert the template into each file.
    s=Template(template_data)
    for hole_file in hole_files:
        for enable, gravity_file_str in gravity_options:
            experiment_data = s.substitute(hole_filename=hole_file, gravity_enable=enable)
            hole_id = re.search("id=([0-9]*)",hole_file).group(1)
            experiment_filename = os.path.join(experiment_dir, "full_peg_insertion_experiment{gravity_file_str}_moving_hole_id={hole_id}.xml".format(gravity_file_str=gravity_file_str, hole_id=hole_id)) 
            with open(experiment_filename, 'w') as experiment_file:
                experiment_file.write(experiment_data)

if __name__=='__main__':
    files = get_hole_files()
    gen_experiment_files(files)

import os
import re
from datetime import datetime


def get_experiment_root():
    '''
    Returns the root directory for the experiments as specified by the
    OPENAI_LOGDIR environment variable.
    '''
    return os.path.join(os.environ['OPENAI_LOGDIR'], 'stable')

def new_experiment_dir(params):
    '''
    Generates the path to save the model and the experiment data from a
    dictionary of parameters.
    '''

    # Create a unique path based on the date and time of the experiment.
    day, time = datetime.now().isoformat().split('T')

    # Create a unique path based on a description of the experiment.
    description = ['alg={}'.format(params['alg']), 'env={}'.format(params['env'])]
    for k,v in sorted(params['learning_options'].items()):
        description.append('{}={}'.format(k,v))

    for k,v in sorted(params['actor_options'].items()):
        description.append('{}={}'.format(k,v))
    description = ','.join(description)
    
    # Contruct the path and return.
    save_path = os.path.join(get_experiment_root(), day, time, description)
    return save_path


def get_experiment_dirs(rel_path=''):
    '''
    Searches for all of the experiment directories relative to the root
    directory, or another relative path underneath the root.
    '''
    root = os.path.join(get_experiment_root(), rel_path)

    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'params.json' in filenames:
            dirnames.clear()
            paths.append(dirpath)

    return paths

def get_params(experiment_dir):
    return os.path.join(experiment_dir, 'params.json')

def get_checkpoints(experiment_dir):
    '''
    Returns a list of all of the checkpoints in the experiment directory.
    '''
    return [
        entry.path for entry in os.scandir(experiment_dir)
        if entry.is_file() and 'checkpoint' in entry.name
    ]

def get_model(experiment_dir):
    '''
    Returns the path to the final model if it exists, otherwise returns None.
    '''
    if 'model.pkl' in os.listdir(experiment_dir):
        return os.path.join(experiment_dir, 'model.pkl')


def get_latest_checkpoint(experiment_dir):
    '''
    Gets the latest checkpoint file in the experiment directory.
    '''
    checkpoint_paths = get_checkpoints(experiment_dir)
    assert len(
        checkpoint_paths) > 0, 'There are no checkpoints in this directory.'
    argmax = -1
    maximum = 0
    for i, ckpt_path in enumerate(checkpoint_paths):
        checkpoint_num = int(re.search(r'_(\d+).pkl', ckpt_path).group(1))
        if checkpoint_num > maximum:
            argmax, maximum = i, checkpoint_num

    latest_checkpoint = checkpoint_paths[argmax]
    return latest_checkpoint


if __name__ == '__main__':
    # Test that these functions return something without failing.
    get_experiment_root()
    get_experiment_dirs()
    experiment_dir = get_experiment_dirs(
        rel_path='2019-01-17/08:54:55.092869')[0]
    checkpoints = get_checkpoints(experiment_dir)
    latest_checkpoint = get_latest_checkpoint(experiment_dir)
    print(latest_checkpoint)
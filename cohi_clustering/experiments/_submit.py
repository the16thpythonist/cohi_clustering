import os
import pathlib
import subprocess

PATH: str = pathlib.Path(__file__).parent.absolute()

# This is the config to be used for the AutoSlurm scheduling.
ASLURM_CONFIG: str = 'euler'

VENV_PATH: str = '/media/ssd2/Programming/cohi_clustering/.venv'

# The path to the experiment module to be run.
EXPERIMENT_PATH = os.path.join(PATH, 'contrastive_graphs.py')

# The parameters to be used to run the python script
EXPERIMENT_PARAMETERS = {
    '__DEBUG__': False,
    '__PREFIX__': 'run',
    'NUM_EPOCHS': 40,
}

python_command_list = [
    'python',
    EXPERIMENT_PATH,
    *[
        f'--{key}="{repr(value)}"'
        for key, value in EXPERIMENT_PARAMETERS.items()
    ]
]
python_command = ' '.join(python_command_list)

result = subprocess.run([
    'aslurmx', 
    '-cn',
    ASLURM_CONFIG,
    '-o',
    f'venv={VENV_PATH}', 
    'cmd',
    python_command,
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print('STDOUT:', result.stdout.decode())
print('STDERR:', result.stderr.decode())
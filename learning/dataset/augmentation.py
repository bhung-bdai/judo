
from dataclasses import dataclass

import mujoco
from judo.simulation import get_simulation_backend
from judo.tasks import Task, TaskConfig,  get_task_config, get_task_registration

@dataclass
class AugmentationConfig:
    task_name: str
    randomization_percentage: float


def augment_dataset(augmentation_config: AugmentationConfig) -> None:
    # Load the task
    task_registration = get_task_registration(augmentation_config.task_name)
    task_type = task_registration.task_type
    task_config_type = task_registration.task_config_type
    rollout_backend = task_registration.rollout_backend
    simulation_backend = task_registration.simulation_backend
    locomotion_policy_path = task_registration.locomotion_policy_path

    sim_backend = get_simulation_backend(simulation_backend)
    sim = sim_backend(init_task=augmentation_config.task_name)



    



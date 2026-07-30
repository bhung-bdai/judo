

from learning.dynamics.tasks.base import Task
from learning.dynamics.tasks.cylinder_push import CylinderPushTask

task_registry = {
    "cylinderpushtask": CylinderPushTask,
}

def resolve_task_name(task_name: str) -> Task:
    lower_name = task_name.lower()
    if lower_name not in task_registry:
        raise ValueError(f"Task {task_name} not found in registry")
    return task_registry[lower_name]

__all__ = ["Task", "CylinderPushTask"]
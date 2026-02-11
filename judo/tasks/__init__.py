# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.spot import (
    SpotBarbellLift,
    SpotBarbellLiftConfig,
    SpotBase,
    SpotBaseConfig,
    SpotBoxPush,
    SpotBoxPushConfig,
    SpotBoxWithHandlePush,
    SpotBoxWithHandlePushConfig,
    SpotHandNavigate,
    SpotHandNavigateConfig,
    SpotNavigate,
    SpotNavigateConfig,
    SpotSmallBoxPush,
    SpotSmallBoxPushConfig,
    SpotTableDrag,
    SpotTableDragConfig,
    SpotTireDrag,
    SpotTireDragConfig,
    SpotTireDrop,
    SpotTireDropConfig,
    SpotTireNavigate,
    SpotTireNavigateConfig,
    SpotTireRackInsert,
    SpotTireRackInsertConfig,
    SpotTireRoll,
    SpotTireRollConfig,
    SpotTireStack,
    SpotTireStackConfig,
    SpotTireUpright,
    SpotTireUprightConfig,
    SpotWheelRimRoll,
    SpotWheelRimRollConfig,
    SpotYellowChairPush,
    SpotYellowChairPushConfig,
    SpotYellowChairRampPush,
    SpotYellowChairRampPushConfig,
)

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    CylinderPush.name: (CylinderPush, CylinderPushConfig),
    Cartpole.name: (Cartpole, CartpoleConfig),
    FR3Pick.name: (FR3Pick, FR3PickConfig),
    LeapCube.name: (LeapCube, LeapCubeConfig),
    LeapCubeDown.name: (LeapCubeDown, LeapCubeDownConfig),
    CaltechLeapCube.name: (CaltechLeapCube, CaltechLeapCubeConfig),
    SpotBase.name: (SpotBase, SpotBaseConfig),
    SpotBarbellLift.name: (SpotBarbellLift, SpotBarbellLiftConfig),
    SpotBoxPush.name: (SpotBoxPush, SpotBoxPushConfig),
    SpotBoxWithHandlePush.name: (SpotBoxWithHandlePush, SpotBoxWithHandlePushConfig),
    SpotHandNavigate.name: (SpotHandNavigate, SpotHandNavigateConfig),
    SpotNavigate.name: (SpotNavigate, SpotNavigateConfig),
    SpotSmallBoxPush.name: (SpotSmallBoxPush, SpotSmallBoxPushConfig),
    SpotTableDrag.name: (SpotTableDrag, SpotTableDragConfig),
    SpotTireDrag.name: (SpotTireDrag, SpotTireDragConfig),
    SpotTireDrop.name: (SpotTireDrop, SpotTireDropConfig),
    SpotTireNavigate.name: (SpotTireNavigate, SpotTireNavigateConfig),
    SpotTireRackInsert.name: (SpotTireRackInsert, SpotTireRackInsertConfig),
    SpotTireRoll.name: (SpotTireRoll, SpotTireRollConfig),
    SpotTireStack.name: (SpotTireStack, SpotTireStackConfig),
    SpotTireUpright.name: (SpotTireUpright, SpotTireUprightConfig),
    SpotWheelRimRoll.name: (SpotWheelRimRoll, SpotWheelRimRollConfig),
    SpotYellowChairPush.name: (SpotYellowChairPush, SpotYellowChairPushConfig),
    SpotYellowChairRampPush.name: (SpotYellowChairRampPush, SpotYellowChairRampPushConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "SpotBarbellLift",
    "SpotBarbellLiftConfig",
    "SpotBase",
    "SpotBaseConfig",
    "SpotBoxPush",
    "SpotBoxPushConfig",
    "SpotBoxWithHandlePush",
    "SpotBoxWithHandlePushConfig",
    "SpotHandNavigate",
    "SpotHandNavigateConfig",
    "SpotNavigate",
    "SpotNavigateConfig",
    "SpotSmallBoxPush",
    "SpotSmallBoxPushConfig",
    "SpotTableDrag",
    "SpotTableDragConfig",
    "SpotTireDrag",
    "SpotTireDragConfig",
    "SpotTireDrop",
    "SpotTireDropConfig",
    "SpotTireNavigate",
    "SpotTireNavigateConfig",
    "SpotTireRackInsert",
    "SpotTireRackInsertConfig",
    "SpotTireRoll",
    "SpotTireRollConfig",
    "SpotTireStack",
    "SpotTireStackConfig",
    "SpotTireUpright",
    "SpotTireUprightConfig",
    "SpotWheelRimRoll",
    "SpotWheelRimRollConfig",
    "SpotYellowChairPush",
    "SpotYellowChairPushConfig",
    "SpotYellowChairRampPush",
    "SpotYellowChairRampPushConfig",
]

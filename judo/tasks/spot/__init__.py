# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Spot locomotion and manipulation tasks."""

from judo.tasks.spot.spot_barbell_lift import SpotBarbellLift, SpotBarbellLiftConfig
from judo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from judo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from judo.tasks.spot.spot_box_with_handle_push import SpotBoxWithHandlePush, SpotBoxWithHandlePushConfig
from judo.tasks.spot.spot_hand_navigate import SpotHandNavigate, SpotHandNavigateConfig
from judo.tasks.spot.spot_navigate import SpotNavigate, SpotNavigateConfig
from judo.tasks.spot.spot_small_box_push import SpotSmallBoxPush, SpotSmallBoxPushConfig
from judo.tasks.spot.spot_table_drag import SpotTableDrag, SpotTableDragConfig
from judo.tasks.spot.spot_tire_drag import SpotTireDrag, SpotTireDragConfig
from judo.tasks.spot.spot_tire_drop import SpotTireDrop, SpotTireDropConfig
from judo.tasks.spot.spot_tire_navigate import SpotTireNavigate, SpotTireNavigateConfig
from judo.tasks.spot.spot_tire_rack_insert import SpotTireRackInsert, SpotTireRackInsertConfig
from judo.tasks.spot.spot_tire_roll import SpotTireRoll, SpotTireRollConfig
from judo.tasks.spot.spot_tire_stack import SpotTireStack, SpotTireStackConfig
from judo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig
from judo.tasks.spot.spot_wheel_rim_roll import SpotWheelRimRoll, SpotWheelRimRollConfig
from judo.tasks.spot.spot_yellow_chair_push import SpotYellowChairPush, SpotYellowChairPushConfig
from judo.tasks.spot.spot_yellow_chair_ramp_push import SpotYellowChairRampPush, SpotYellowChairRampPushConfig

__all__ = [
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

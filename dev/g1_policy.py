# Copyright (c) 2026 Robotics and AI Institute LLC. All rights reserved.

import mujoco

from judo import MODEL_PATH

XML_PATH = str(MODEL_PATH / "xml/g1/g1_flat.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

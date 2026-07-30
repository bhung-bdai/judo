<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Summary: Dataset spot_tire_upright3 Components

  ## 📊 QPOS (33 components)

  Robot Base (7 - free joint):
  0-2:   spot/base.{x, y, z}           # Position in world frame
  3-6:   spot/base.{qw, qx, qy, qz}    # Orientation quaternion

  ### Leg Joints (12 - 4 legs × 3 joints):
   7-9:  spot/fl_{hx, hy, kn}          # Front-left: hip-x, hip-y, knee
  10-12: spot/fr_{hx, hy, kn}          # Front-right
  13-15: spot/hl_{hx, hy, kn}          # Hind-left
  16-18: spot/hr_{hx, hy, kn}          # Hind-right

  Arm Joints (7):
  19-25: spot/arm_{sh0, sh1, el0, el1, wr0, wr1, f1x}
         # Shoulder(2), Elbow(2), Wrist(2), Finger(1)

  Object (Tire) (7 - free joint):
  26-28: object/tire_joint.{x, y, z}   # Position in world frame
  29-32: object/tire_joint.{qw, qx, qy, qz}  # Orientation quaternion

  ---
  ## 🏃 QVEL (31 components)

  Robot Base Velocity (6):
  0-2:  spot/base.{vx, vy, vz}         # Linear velocity (world frame)
  3-5:  spot/base.{wx, wy, wz}         # Angular velocity (body frame)

  Leg Joint Velocities (12):
   6-8:  spot/fl_{hx, hy, kn}
   9-11: spot/fr_{hx, hy, kn}
  12-14: spot/hl_{hx, hy, kn}
  15-17: spot/hr_{hx, hy, kn}

  Arm Joint Velocities (7):
  18-24: spot/arm_{sh0, sh1, el0, el1, wr0, wr1, f1x}

  Object Velocity (6):
  25-27: object/tire_joint.{vx, vy, vz}  # Linear velocity (world frame)
  28-30: object/tire_joint.{wx, wy, wz}  # Angular velocity (body frame)

  ---
  ## 🎮 ACTION (17 components)

  Leg Actuators (12):
   0-2:  spot/act_{fl1, fl2, fl3}      # Front-left leg
   3-5:  spot/act_{fr1, fr2, fr3}      # Front-right leg
   6-8:  spot/act_{rl1, rl2, rl3}      # Rear-left leg
   9-11: spot/act_{rr1, rr2, rr3}      # Rear-right leg

  Arm Actuators (5):
  12-16: spot/act_{0, 1, 2, 3, 4, 5, 6}  # Arm joints

  Note: Dataset has 17 actions vs 19 model actuators. The 2 missing are likely gripper finger
  actuators not used in the task.

  ---
  ## 📡 SENSORS (43 components)

  Position Sensors (12):
  - Body, Gripper, Front-left leg, Front-right leg positions (4 × 3D = 12)

  Orientation Sensors (15):
  - Body, Gripper, Finger, Object x/y/z axes (5 × 3D = 15)

  Distance Sensors (4):
  - Leg-to-object distances (fl, fr, rl, rr)

  Other Position Sensors (12):
  - Body, Arm-finger link, Object positions and axes (4 × 3D = 12)

  ---
  ## 📐 Key Dimensions

  qpos:    (1,004,000, 33)  - 33D state vector
  qvel:    (1,004,000, 31)  - 31D velocity vector
  action:  (1,004,000, 17)  - 17D action vector
  sensors: (1,004,000, 43)  - 43D sensor vector

  This matches the Spot robot with tire manipulation task structure!

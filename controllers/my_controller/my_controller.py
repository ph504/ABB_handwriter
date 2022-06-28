# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstration of inverse kinematics using the "ikpy" Python module."""
#############################################

#Libraries
import sys
import tempfile
import pandas as pd
try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

import math
from controller import Supervisor

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')
#############################################


# Constants
IKPY_MAX_ITERATIONS = 4

NO_DIGITS = 10 # number of digits

data_dir = '..\\..\\data\\path t numbers\\'
data_fname = 'patht_Num_'

DRAWING_TIME = 2 * math.pi + 1.5
DIGIT = 2
#############################################

# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
armChain = Chain.from_urdf_file(filename)
for i in [0, 6]:
    armChain.active_links_mask[i] = False

# Load xt and yt data.
df = []
for i in range(NO_DIGITS):
    df.append(pd.read_csv(data_dir+
        data_fname+str(i)+'.csv'))

# Initialize the arm motors and encoders.
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# Get the arm and target nodes.
target = supervisor.getFromDef('TARGET')
arm = supervisor.getSelf()

# Loop 1: Draw the digit on the paper sheet.
print('Draw the digit on the paper sheet...')
ctr = 0
clock_period = DRAWING_TIME/len(df[DIGIT])
clk_start = supervisor.getTime()

while supervisor.step(timeStep) != -1:
    # t = supervisor.getTime()

    # Use the circle equation relatively to the arm base as an input of the IK algorithm.
    # x = 0.25 * math.cos(t) + 1.1# + 0.02*t
    # y = 0.25 * math.sin(t) - 0.95# + 0.02*t

    x = df[DIGIT]['x'].tolist()[ctr]
    y = df[DIGIT]['y'].tolist()[ctr]
    # print('x:', x, 'y:', y)
        
    z = 0.05

    # Call "ikpy" to compute the inverse kinematics of the arm.
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
    ikResults = armChain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

    # Actuate the 3 first arm motors with the IK results.
    for i in range(3):
        motors[i].setPosition(ikResults[i + 1])
    # Keep the hand orientation down.
    motors[4].setPosition(-ikResults[2] - ikResults[3] + math.pi / 2)
    # Keep the hand orientation perpendicular.
    motors[5].setPosition(ikResults[1])

    # Conditions to start/stop drawing and leave this loop.
    
    if supervisor.getTime() > 1.5:
        # Note: start to draw at 1.5 second to be sure the arm is well located.
        if(supervisor.getTime() - clk_start > clock_period):
            ctr +=1
            clk_start = supervisor.getTime()

        if ctr > len(df[DIGIT])-1:
            break
        supervisor.getDevice('pen').write(True)

# Loop 2: Move the arm hand to the target.
print('Move the yellow and black sphere to move the arm...')
while supervisor.step(timeStep) != -1:
    supervisor.getDevice('pen').write(True)
    # Get the absolute postion of the target and the arm base.
    targetPosition = target.getPosition()
    armPosition = arm.getPosition()

    # Compute the position of the target relatively to the arm.
    # x and y axis are inverted because the arm is not aligned with the Webots global axes.
    x = -(targetPosition[1] - armPosition[1])
    y = targetPosition[0] - armPosition[0]
    z = targetPosition[2] - armPosition[2]

    # Call "ikpy" to compute the inverse kinematics of the arm.
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
    ikResults = armChain.inverse_kinematics([x, y, z], max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)

    # Recalculate the inverse kinematics of the arm if necessary.
    position = armChain.forward_kinematics(ikResults)
    squared_distance = (position[0, 3] - x)**2 + (position[1, 3] - y)**2 + (position[2, 3] - z)**2
    if math.sqrt(squared_distance) > 0.03:
        ikResults = armChain.inverse_kinematics([x, y, z])

    # Actuate the arm motors with the IK results.
    for i in range(len(motors)):
        motors[i].setPosition(ikResults[i + 1])

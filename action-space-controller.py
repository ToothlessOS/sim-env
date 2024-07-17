# Action space controller for franka in Isaac Sim
# With APIs for RGBD cameras and kinematics data
### IMPORTANT ###
FLAG = 1 # 0 for default, 1 for test

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Simulating desktop manipulation tasks.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.actuators import IdealPDActuatorCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.types import ArticulationActions

@configclass
class TableTopSceneConfig(InteractiveSceneCfg):
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation (from the source cfg of omni.isaac.lab_assets.FRANKA_PANDA_HIGH_PD_CFG)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            # change config for the gripper here
            "panda_hand": IdealPDActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3, #2e3
                damping=1e2, #1e2
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    # sensor: global camera
    camera = CameraCfg(
        prim_path="/World/Robot/base/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    
    # add custom cameras below:

    # add custom scene objs (e.g. another robot or sensor?) below:


def stream_camera(scene: InteractiveScene, camera_name: str):
    while simulation_app.is_running():
        print(scene[camera_name])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)

def stream_kinematics():
    pass

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # print(robot) -- Type: Articulation
    
    # Create inverse kinematics controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers (for visualization)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # initialize action-space commands queue
    # [x, y, z, rotation in quaternion, gripper opening width]
    commands_queue = [[0.5, 0.5, 0.7, 0.707, 0, 0.707, 0, 0.0]] # init
    # for testing:
    test_commands_queue = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0, 0.0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    commands_queue = torch.tensor(commands_queue, device=sim.device) # FLAG = 0
    test_commands_queue = torch.tensor(test_commands_queue, device=sim.device) # FLAG = 1
    print(len(test_commands_queue))

     # Track the given command
    current_goal_idx = 0

    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    gripper_commands = torch.zeros(scene.num_envs, 1, device=robot.device)
    if FLAG == 1:
        ik_commands[:] = test_commands_queue[:, :-1][current_goal_idx]
        gripper_commands[:] = test_commands_queue[:, -1][current_goal_idx]
    elif FLAG == 0:
        ik_commands[:] = commands_queue[:, :-1][current_goal_idx]
        gripper_commands[:] = commands_queue[:, -1][current_goal_idx]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]) # Include gripper joints but not for IK?
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Key: Queue/stack control?
        if FLAG == 1: # Testing
            # reset
            if count % 150 == 0:
                # reset time
                count = 0
                # reset joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
                # reset actions
                ik_commands[:] = test_commands_queue[:, :-1][current_goal_idx]
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                # gripper_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone() #Bug: should refer to gripper fingers
                # reset controller
                diff_ik_controller.reset()
                diff_ik_controller.set_command(ik_commands)
                # reset grippter actuator
                # change goal
                current_goal_idx = (current_goal_idx + 1) % len(test_commands_queue[:, :-1])
            else:
                # obtain quantities from simulation
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                root_pose_w = robot.data.root_state_w[:, 0:7]
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

                # gripper control (get the actuator from the articulation)
                # https://isaac-sim.github.io/IsaacLab/source/api/lab/omni.isaac.lab.actuators.html#omni.isaac.lab.actuators.IdealPDActuator.compute
                # A slider joint: param: a scalar
                gripper_target = ArticulationActions(
                    joint_positions = torch.tensor([0.01, 0.01], device=sim.device),
                    joint_velocities = torch.tensor([0, 0], device=sim.device),
                    joint_efforts = torch.tensor([0, 0], device=sim.device)
                ) # At this stage: change the target here.
                gripper_pos = robot.data.joint_pos[0, -2:]
                print("POS: " + str(gripper_pos))
                gripper_vel = robot.data.joint_vel[0, -2:]
                print("VEL: " + str(gripper_vel))
                gripper_pos_des = robot.actuators["panda_hand"].compute(gripper_target, gripper_pos, gripper_vel)
        elif FLAG == 0: # With real-time interactions
            pass

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        # robot.set_joint_position_target(gripper_pos_des, joint_ids=robot_entity_cfg.joint_ids) #Bug: should refer to gripper fingers
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    # design_scene()
    scene_cfg = TableTopSceneConfig(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__=="__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
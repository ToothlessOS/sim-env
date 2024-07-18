# Action space controller for franka in Isaac Sim
# With APIs for RGBD cameras and kinematics data
# Version 1: Implemented with Manager-Based base environment

# App launcher
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
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.types import ArticulationActions

# Part 1: Interactive scene def
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

# Part 2: Action def
@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # Note: MDP = Markov Decision Process
    
    # manipulator actions
    joint_efforts = mdp.JointPositionActionCfg(asset_name="robot", joint_names="panda_joint.*")
    # gripper actions
    gripper_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names="panda_finger_joint.*")
    gripper_joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names="panda_finger_joint.*")
    gripper_joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names="panda_finger_joint.*")

# Part 3: Observations def
@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class ManipulatorCfg(ObsGroup):
        """obeservation specifications for the manipulator."""
         # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos) # Absolute joint pos
        joint_vel_rel = ObsTerm(func=mdp.joint_vel) # Ref: v0 Line 254 actuator input

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    class GripperCfg(ObsGroup):
        """obeservation specifications for the gripper."""
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos) # Absolute joint pos
        joint_vel_rel = ObsTerm(func=mdp.joint_vel) # Ref: v0 Line 254 actuator input

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    manipulator_policy: ManipulatorCfg = ManipulatorCfg()
    gripper_policy: GripperCfg = GripperCfg()
        
# Part 4: Events def
@configclass
class EventCfg:
    # On sim startup:
    sim_setup = EventTerm(

    )
    # On sim reset:

    # Periodic:
    pass

# Part 5: Global env def
class ManipulatorEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""
    # scene settings
    scene = TableTopSceneConfig(num_envs=1024, env_spacing=2.5)
    # env settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    # sim settings
    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [[2.5, 2.5, 2.5]]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


# Main function
def main():
    """Main function."""
    # parse the arguments
    env_cfg = ManipulatorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # update counter
            count += 1

    # close the environment
    env.close()

# Utils and APIs
def stream_camera(scene: InteractiveScene, camera_name: str):
    while simulation_app.is_running():
        print(scene[camera_name])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)

def stream_kinematics():
    pass

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

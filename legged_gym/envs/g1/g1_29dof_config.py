from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1_29DOF_RoughCfg( LeggedRobotCfg ):
    class env:
        num_prop = 105
        num_scan = 154
        num_observations = num_prop + num_scan
        num_actions = 29

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint' : -0.1,
            'left_hip_roll_joint' : 0,
            'left_hip_yaw_joint' : 0.,
            'left_knee_joint' : 0.3,
            'left_ankle_pitch_joint' : -0.2,
            'left_ankle_roll_joint' : 0,
            'right_hip_pitch_joint' : -0.1,
            'right_hip_roll_joint' : 0,
            'right_hip_yaw_joint' : 0.,
            'right_knee_joint' : 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint' : 0,

            'waist_yaw_joint': 0,
            'waist_roll_joint': 0,
            'waist_pitch_joint': 0,

            'left_shoulder_pitch_joint': 0.,
            'left_shoulder_roll_joint': 0.,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_joint': 0., 
            'left_wrist_roll_joint': 0,
            'left_wrist_pitch_joint': 0,
            'left_wrist_yaw_joint': 0,
            'right_shoulder_pitch_joint': 0.,
            'right_shoulder_roll_joint': 0.,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_joint': 0.,
            'right_wrist_roll_joint': 0,
            'right_wrist_pitch_joint': 0,
            'right_wrist_yaw_joint': 0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_pitch': 100,
                     'hip_roll': 120,
                     'hip_yaw': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 200,
                     'shoulder': 50,
                     'elbow': 50,
                     'wrist': 40,
                     }  # [N*m/rad]
        damping = {  'hip_pitch': 2,
                     'hip_roll': 3,
                     'hip_yaw': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 5,
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        shoulder_coupling_gain = 1

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_rev_1_0.urdf'
        name = "g1_29dof"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        hip_dof_name = ["hip_roll", "hip_yaw"]
        hip_pitch_dof_name = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
        shoulder_dof_name = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]

class G1_29DOF_RoughCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000
        run_name = ''
        experiment_name = 'g1_29dof'
        save_interval = 5000 # check for potential saves every this many iterations
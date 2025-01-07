from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HustBotRoughCfg( LeggedRobotCfg ):

    class env(LeggedRobotCfg.env):
        num_observations = 41
        num_privileged_obs = 44
        num_actions = 10
        num_envs = 4096
        episode_length_s = 24

    
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hustBot2.5/urdf/hustBot2_5.urdf'
        name = "hustBot"
        foot_name = "foot"
        knee_name = "knee"
        penalize_contacts_on = ["base_link"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.7, 0.0, 0.3, 0.0, 0.0, 0, 0]
        restitution = 0.
    
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 2.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.15
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.92]
        
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'right_hip_roll_joint': 0.,
            'right_hip_pitch_joint': -0.25,
            'right_knee_joint': 0.45,
            'right_ankle_joint': -0.2,
            'right_foot_joint': 0.,
            'left_hip_roll_joint': 0.,
            'left_hip_pitch_joint': -0.25,
            'left_knee_joint': 0.45,
            'left_ankle_joint': -0.2,
            'left_foot_joint': 0.,
        }
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        # hustBot 机器人配置
        stiffness = {'hip_roll': 450, 'hip_pitch': 60, 'knee': 1000,
                     'ankle': 20, 'foot': 20}
        damping = {'hip_roll': 4, 'hip_pitch': 4, 'knee':
                   30, 'ankle': 2, 'foot': 2}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-4., 4.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      
    class domain_rand(LeggedRobotCfg.domain_rand):
        # 摩擦力随机化
        randomize_friction = True
        friction_range = [0.5, 1.25]
        # 推力随机化
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        # 基座质量随机化
        randomize_base_mass = True
        added_mass_range = [-1., 1.]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.7

        num_commands = 4
        resampling_time = 6.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        # class ranges:
        #     lin_vel_x = [-0.5, 1.5] # min max [m/s] 
        #     lin_vel_y = [-0.5, 0.5]   # min max [m/s]
        #     ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
        #     heading = [-3.14, 3.14]

        class ranges:
            lin_vel_x = [-0.5, 0.8]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-0.0, 0.0]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.88
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

class HustBotRoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        # actor_hidden_dims = [32]
        # critic_hidden_dims = [32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'hustBot'

  

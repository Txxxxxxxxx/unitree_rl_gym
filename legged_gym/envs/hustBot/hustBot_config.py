from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class HustBotCfg(LeggedRobotCfg):
    """
    Configuration class for the zqsa01 humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 41
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 68
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 4096
        episode_length_s = 24 # episode length in seconds
        use_ref_actions = False
        joint_num = 10
        
    class safety:
        # safety factors
        pos_limit = 0.95
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hustBot2.5/urdf/hustBot2_5.urdf'

        name = "hustBot"
        foot_name = "foot"
        knee_name = "knee"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
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

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        # hustBot 机器人配置
        stiffness = {'hip_roll': 450, 'hip_pitch': 60, 'knee': 1000,
                     'ankle': 20, 'foot': 20}
        damping = {'hip_roll': 4, 'hip_pitch': 4, 'knee':
                   30, 'ankle': 2, 'foot': 2}

        # stiffness = {'1_joint': 50, '2_joint': 50,'3_joint': 70,
        #              '4_joint': 70, '5_joint': 20, '6_joint': 20}
        # damping = {'1_joint': 5.0, '2_joint': 5.0,'3_joint': 7.0, 
        #            '4_joint': 7.0, '5_joint': 2, '6_joint': 2}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

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
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.6
        max_push_ang_vel = 0.8

        randomize_base_mass = True
        added_base_mass_range = [-4.0, 4.0]

        randomize_base_com = True
        added_base_com_range = [-0.1, 0.1]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]    

        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]

        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]

        randomize_joint_armature = True
        joint_armature_range = [0.008, 0.06]    #

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 10]
        
        add_obs_latency = True # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 10]
        range_obs_imu_latency = [1, 10]

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

    class rewards:
        stand_still_flag = True
        stand_still_hold = 0.05
        stand_still_prop = 0.3
        base_height_target = 0.88
        min_dist = 0.2
        max_dist = 0.8
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.18    # rad ?
        target_feet_height = 0.03       # m  ?
        cycle_time = 0.64                # sec ??
        only_positive_rewards = True
        tracking_sigma = 15
        max_contact_force = 400  # forces above this value are penalized
        
        class scales:
            joint_pos = 1.1
            feet_clearance = 1.0
            feet_contact_number = 1.0
            # gait
            feet_air_time = 1.3
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            # contact 
            feet_contact_forces = -0.005
            feet_contact_forces_distance = -0.005
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # stand_still = 5
            # base pos
            default_joint_pos = 0.8
            orientation = 1.
            # orientation_vel = 1.0
            base_height = 0.5
            base_acc = 0.2
            # energy
            action_smoothness = -0.01
            knee_action_smoothness = -0.04
            ankle_action_smoothness = -0.04
            torques = -1e-7
            dof_vel = -8e-5
            dof_acc = -5e-9
            collision = -1.
            # stumble = 1.


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


class HustBotCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        sym_loss = True
        obs_permutation = [-0.0001, -1, 2, -3, -4,\
                           -10, 11, 12, 13, -14, -5 , 6 , 7 , 8 , -9 ,\
                           -20, 21, 22, 23, -24, -15, 16, 17, 18, -19,\
                           -30, 31, 32, 33, -34, -25, 26, 27, 28, -29,\
                           -35, 36, -37, -38, 39, -40]
        act_permutation = [-5, 6, 7, 8, -9, -0.0001, 1, 2, 3, -4,]
        frame_stack = 15  # 
        sym_coef = 1.0


    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HustBot_PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 30001  # number of policy updates

        # logging
        save_interval = 250  # check for potential saves every this many iterations
        experiment_name = 'hustbot_ppo'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    
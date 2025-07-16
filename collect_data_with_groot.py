import argparse
import robosuite
import imageio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from robosuite import make
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply, quat_inverse
from robosuite.controllers import load_composite_controller_config

from copy import deepcopy
import json 

#GR00T Imports
import os
import gr00t
import torch
import tyro
import matplotlib
from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag
#from gr00t.model.gr00t_n1 import GR00T_N1 #outdated
from gr00t.utils.peft import get_lora_model
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from transformers import TrainingArguments


from scipy.spatial.transform import Rotation as R


def prepare_data_for_gr00t(prompt, image_frames, states):
    #For GR00T we need the LeRobot data, so we will follow the procedures mentioned in the  documentation (https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/LeRobot_compatible_data_schema.md)
    
    #the dataset for one prediction is a dictionary
    #Dataset we need for get_action: video.ego_view, state.left_arm, state.right_arm, state.left_hand, state.right_hand, action.right_arm, action.left_arm, action.left_hand, action.right_hand, annotation.human.task_description
    pass
def extract_gr00t_data_from_obs(obs, env):
    #For the hand, we use the robot0_right_gripper_qpos
    #full_gripper_vec_right=obs["robot0_right_gripper_qpos"]
    #full_gripper_vec_left=obs["robot0_left_gripper_qpos"]
    
    qpos_indices_right_hand=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    qpos_indices_left_hand=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    full_gripper_vec_right=np.array(env.sim.data.qpos[qpos_indices_right_hand])
    #print("Full gripper right: ", full_gripper_vec_right)

    full_gripper_vec_left=np.array(env.sim.data.qpos[qpos_indices_left_hand])

    #From this, I follow the robocasa simulation example (https://github.com/robocasa/robocasa-gr1-tabletop-tasks/blob/main/robocasa/models/robots/__init__.py#L101)
    right_hand = full_gripper_vec_right[[0, 1, 4, 6, 8, 10]][::-1] #now this is somehow also reversed, I dont know why, and Im not sure if I will
    #print("Right hand state: ", right_hand)
    left_hand = full_gripper_vec_left[[0, 1, 4, 6, 8, 10]][::-1]




    #After that we need the arms states
    #I am trying this now IT MIGHT NOT BE RIGHT!!!!!!!
    #right_arm=obs["robot0_joint_pos"][7:]
    #left_arm=obs["robot0_joint_pos"][:7]
    #I will do this differently now to try to make sure we are "attacking" the right joints
    all_joints_right=["robot0_r_shoulder_pitch", 
                      "robot0_r_shoulder_roll",
                      "robot0_r_shoulder_yaw",
                      "robot0_r_elbow_pitch",
                      "robot0_r_wrist_yaw",
                      "robot0_r_wrist_roll",
                      "robot0_r_wrist_pitch"]

    all_joints_left=["robot0_l_shoulder_pitch", 
                      "robot0_l_shoulder_roll",
                      "robot0_l_shoulder_yaw",
                      "robot0_l_elbow_pitch",
                      "robot0_l_wrist_yaw",
                      "robot0_l_wrist_roll",
                      "robot0_l_wrist_pitch"]
    #all_joint_idx_right=[env.sim.model.joint_name2id(el) for el in all_joints_right]
    right_arm = [env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name)] for name in all_joints_right]
    left_arm = [env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(name)] for name in all_joints_left]

    #print("Right hand state: ", right_arm)

    #So it will be: left_arm, left_hand, left_leg, neck, right_arm, right_hand, right_leg, waist -> they are all zeros in acton and states
    leg=np.array([0., 0., 0., 0., 0., 0.])
    neck=np.array([0., 0., 0.])
    waist=np.array([0., 0., 0.])

    #print("right hand shape: ", six_d_state_of_gripper_right.shape)
    #print("left hand shape: ", six_d_state_of_gripper_left.shape)
    #print("right arm shape: ", right_arm.shape)
    #print("left arm shape: ", left_arm.shape)
    #Lastly, we concat all the arrays into one long array that can be read out with the modality.json file
    all_states=np.concatenate((left_arm, left_hand, leg, neck, right_arm, right_hand, leg, waist))

    video_frame=obs["robot0_robotview_image"]

    return left_arm, right_arm, left_hand, right_hand, video_frame

def joint_pos_to_ee_pose(sim, robot, joint_names, joint_positions, ee_site_name):
    # Set joint positions
    for jname, jval in zip(joint_names, joint_positions):
        jidx = sim.model.joint_name2id(jname)
        qpos_addr = sim.model.jnt_qposadr[jidx]
        sim.data.qpos[qpos_addr] = jval
    
    # Forward kinematics update
    sim.forward()

    # Get position and rotation of EE site
    site_id = sim.model.site_name2id(ee_site_name)
    pos = sim.data.site_xpos[site_id]
    mat = sim.data.site_xmat[site_id].reshape(3, 3)

    # Convert rotation matrix to axis-angle
    rotvec = Rotation.from_matrix(mat).as_rotvec()

    return np.concatenate([pos, rotvec])


def get_axis_angle_actions(old_jp_arms,  old_jp_right_hand, old_jp_left_hand, new_jp_arms, new_jp_right_hand, new_jp_left_hand, env):
    #OLD VALUES
    ee_pos_arms_right_old = env.robots[0].sim.data.get_body_xpos(env.robots[0].robot_model.eef_name["right"]).copy()
    ee_quat_arms_right_old = env.robots[0].sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["right"]).copy()
    axis_angle_arm_right_old = robosuite.utils.transform_utils.quat2axisangle(ee_quat_arms_right_old)
    

    ee_pos_arms_left_old = env.robots[0].sim.data.get_body_xpos(env.robots[0].robot_model.eef_name["left"]).copy()
    ee_quat_arms_left_old = env.robots[0].sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["left"]).copy()
    axis_angle_arm_left_old = robosuite.utils.transform_utils.quat2axisangle(ee_quat_arms_left_old)

    
    ee_pos_hand_right_old = env.robots[0].sim.data.get_body_xpos("gripper0_right_eef").copy()
    ee_quat_hand_right_old = env.robots[0].sim.data.get_body_xquat("gripper0_right_eef").copy()
    axis_angle_hand_right_old = robosuite.utils.transform_utils.quat2axisangle(ee_quat_hand_right_old)

    ee_pos_hand_left_old = env.robots[0].sim.data.get_body_xpos("gripper0_left_eef").copy()
    ee_quat_hand_left_old = env.robots[0].sim.data.get_body_xquat("gripper0_left_eef").copy()
    axis_angle_hand_left_old = robosuite.utils.transform_utils.quat2axisangle(ee_quat_hand_left_old)

    #NEW VALUES
    env.robots[0].set_robot_joint_positions(new_jp_arms)
    env.robots[0].set_gripper_joint_positions(new_jp_right_hand, "right")
    env.robots[0].set_gripper_joint_positions(new_jp_left_hand, "left")

    #print(env.robots[0].sim.data)
    #print(print(env.robots[0].robot_model.eef_name["right"]))
    ee_pos_arms_right = env.robots[0].sim.data.get_body_xpos(env.robots[0].robot_model.eef_name["right"]).copy()
    ee_quat_arms_right = env.robots[0].sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["right"]).copy()
    axis_angle_arm_right = robosuite.utils.transform_utils.quat2axisangle(ee_quat_arms_right)
    

    ee_pos_arms_left = env.robots[0].sim.data.get_body_xpos(env.robots[0].robot_model.eef_name["left"]).copy()
    ee_quat_arms_left = env.robots[0].sim.data.get_body_xquat(env.robots[0].robot_model.eef_name["left"]).copy()
    axis_angle_arm_left = robosuite.utils.transform_utils.quat2axisangle(ee_quat_arms_left)

    #print(env.robots[0].robot_model.grippers)
    ee_pos_hand_right = env.robots[0].sim.data.get_body_xpos("gripper0_right_eef").copy()
    ee_quat_hand_right = env.robots[0].sim.data.get_body_xquat("gripper0_right_eef").copy()
    axis_angle_hand_right = robosuite.utils.transform_utils.quat2axisangle(ee_quat_hand_right)

    ee_pos_hand_left = env.robots[0].sim.data.get_body_xpos("gripper0_left_eef").copy()
    ee_quat_hand_left = env.robots[0].sim.data.get_body_xquat("gripper0_left_eef").copy()
    axis_angle_hand_left = robosuite.utils.transform_utils.quat2axisangle(ee_quat_hand_left)

    #Pos
    delta_arm_right_pos=ee_pos_arms_right-ee_pos_arms_right_old
    delta_arm_left_pos=ee_pos_arms_right_old-ee_pos_arms_left_old
    delta_hand_right_pos=ee_pos_hand_right-ee_pos_hand_right_old
    delta_hand_left_pos=ee_pos_hand_left-ee_pos_hand_left_old

    #Quat
    delta_arm_right_quat=quat2axisangle(quat_multiply(ee_quat_arms_right, quat_inverse(ee_quat_arms_right_old)))
    delta_arm_left_quat=quat2axisangle(quat_multiply(ee_quat_arms_left, quat_inverse(ee_quat_arms_left_old)))
    delta_hand_right_quat=quat2axisangle(quat_multiply(ee_quat_hand_right, quat_inverse(ee_quat_hand_right_old)))
    delta_hand_left_quat=quat2axisangle(quat_multiply(ee_quat_hand_left, quat_inverse(ee_quat_hand_left_old)))


    #After collecing all we need, we reset to the old position
    env.robots[0].set_robot_joint_positions(old_jp_arms)
    env.robots[0].set_gripper_joint_positions(old_jp_right_hand, "right")
    env.robots[0].set_gripper_joint_positions(old_jp_left_hand, "left")
    #print("Delta fingers pos : ", delta_hand_right_pos)
    #print("Delta fingers quat : ", delta_hand_right_quat)
    #return np.concatenate([delta_arm_left_pos, delta_arm_left_quat, 
    #                      delta_arm_right_pos, delta_arm_right_quat, 
    #                      delta_hand_left_pos, delta_hand_left_quat, 
    #                      delta_hand_right_pos, delta_hand_right_quat])
    return np.concatenate([delta_arm_right_pos, delta_arm_right_quat, 
                          delta_arm_left_pos, delta_arm_left_quat, 
                          delta_hand_right_pos, delta_hand_right_quat, 
                          delta_hand_left_pos, delta_hand_left_quat])
    
def convert_training_data_into_right_side_states(df):
    #print(df.keys())
    all_states=df["observation.state"].tolist()

    all_right_arms=[]
    all_right_hands=[]

    for states_list in all_states:
        all_right_arms.append(states_list[22:29])
        all_right_hands.append(states_list[29:35])
    
    print("Hands size: ", len(all_right_hands[0]))
    print("Arm size: ", len(all_right_arms[0]))

    return all_right_arms, all_right_hands

def create_environment(args):
    #env = make( 
    #    args.environment,
    #    args.robots,
    #    has_renderer=False,
    #    has_offscreen_renderer=True, 
    #    ignore_done=True, 
    #    use_camera_obs=True,
    #    control_freq=20,
    #    camera_names=args.camera,
    #    camera_heights=args.height,
    #    camera_widths=args.width)
    #controller_configs = load_composite_controller_config(
    #        controller=None,
    #        robot=args.robots,
    #    )
    #controller_configs["type"] = "BASIC"
    #controller_configs["composite_controller_specific_configs"] = {}
    #controller_configs["control_delta"] = True

    states_hands_right=[
        [0.003, 0.003, 0., 0.,  0.,  0.],
        [0.023, 0.03, 0.003, 0.004,  0.003,  0.003],
        [0.081,  0.031,  0.011,  0.013,  0.011,  0.009],
        [0.143,  0.017,  0.021,  0.025,  0.022,  0.019],
        [0.179,  0.006,  0.03,  0.036,  0.03,   0.026],
        [0.174,  0.013,  0.032, 0.038,  0.032,  0.027],
        [0.834,  0.399,  0.416,  0.425,  0.344,  0.256]]
    

    states_arms_right=[
        [-0.015,-0.077,-0.021,-1.549, 0.013, -0.051, 0.046],
        [-0.018,-0.022, 0.052, -1.563, 0.077, -0.124,  0.065],
        [-0.152, -0., 0.434, -1.398,  0.388, -0.565, 0.212],
        [-0.293,  0., 0.582, -1.254,  0.732, -0.656, 0.509],
        [-0.32,  -0.,  0.61,  -1.241,  0.706, -0.474 , 0.389],
        [-0.314, -0.,  0.598, -1.167,  0.671, -0.431,  0.259],
        [-0.34,  -0.001,  0.522, -0.831,  0.592, -0.593, -0.081]]
    
    states_hands_right=[
        [0.003, 0.0, 0., 0.,  0.,  0.],
        [0.03, 0.0, 0., 0.,  0.,  0.],
        [0.3, 0.0, 0., 0.,  0.,  0.],
        [0.6, 0.0, 0., 0.,  0.,  0.],
        [0.9, 0.0, 0., 0.,  0.,  0.],
        [1.0, 0.0, 0., 0.,  0.,  0.],
        [1.003, 0.0, 0., 0.,  0.,  0.],
        [1.5, 0.0, 0., 0.,  0.,  0.],
        [2.0, 0.0, 0., 0.,  0.,  0.],
        [3.0, 0.0, 0., 0.,  0.,  0.],
    ]
    full_number_of_updates=0

    training_data=pd.read_parquet("finetune_data_pick_up_cube_one_vid_same_pos_new_states/data/chunk-000/episode_000000.parquet")
    states_arms_right, states_hands_right=convert_training_data_into_right_side_states(training_data)


    with open('controller_config.json', 'r') as f:
        controller_configs = json.load(f)

    env = make(
        args.environment,
        args.robots,
        controller_configs=controller_configs,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=80,#100, 
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        horizon= 16,
    )

    
    env.reset()
    
    obs=env._get_observations()
    #print(env.action_spec)
    #print(env.action_dim)
    writer = imageio.get_writer(args.video_path, fps=20)
    frames = []

    #SETUP BEFORE DOING ANY ITERATIONS
    #If we never have an action before, we will just make a zero vector
    actions=np.zeros(44,)
    text=args.prompt
    state_left_arm, state_right_arm, state_left_hand, state_right_hand, video_ego_view=extract_gr00t_data_from_obs(obs, env)
    
    #Experimentally, I set the states of the right arm now:
    state_right_arm=states_arms_right[1] #[-0.152, -0., 0.434, -1.398,  0.388, -0.565,  0.212]
    state_right_hand=states_hands_right[1] #[ 0.081, 0.031,  0.011, 0.013,  0.011,  0.009]

    #This also means I have to make a new "screenshot" of the video, but actually it works okay without that


    action_left_arm=np.zeros((16, 7))#state_left_arm.shape[0]))
    action_right_arm=np.zeros(( 16, 7))#state_right_arm.shape[0]))
    action_left_hand=np.zeros(( 16, 6))#state_left_hand.shape[0]))
    action_right_hand=np.zeros((16, 6))#state_right_hand.shape))
    
    #print("Type video: ", type(video_ego_view[0][0][0]))
    flipped_img =np.flipud(video_ego_view)
    full_dict={"video.ego_view": np.expand_dims(flipped_img, axis=0),
                "state.left_arm": np.array([state_left_arm]),
                "state.right_arm": np.array([state_right_arm]),
                "state.left_hand": np.array([state_left_hand]),
                "state.right_hand": np.array([state_right_hand]),
                
                "action.left_arm": np.array([action_left_arm]),
                "action.right_arm": np.array([action_right_arm]),
                "action.left_hand": np.array([action_left_hand]),
                "action.right_hand": np.array([action_right_hand]),

                "annotation.human.action.task_description": [args.prompt]}

    all_shapes={el[0]:el[1].shape for el in list(full_dict.items()) if type(el[1]) != list}
    #print("Action Spec: ", print(env.action_spec))
    #print("All shapes in collect data: ", all_shapes)
    done=False
    all_actions_plot=[]
    for i in range(120):#len(states_hands_right)):#len(states_hands_right)//16): #range(len(states_hands_right)) for my own demos!
        start = time.time()

        #I COLLECT VIDEO DATA LIKE THIS FOR MY OWN DEMOS REPLAY
        #if i % args.skip_frame == 0:
        #    frame = env.sim.render(camera_name="robot0_robotview", width=256, height=256)
        #    writer.append_data(np.flipud(frame)) 
        #    
            #OLD
            #frame = np.flipud(obs[args.camera + "_image"])#np.flipud(
            #writer.append_data(frame)
        #    print("Saving frame #{}".format(i))

        #action = np.random.randn(*env.action_spec[0].shape) #Now this has to be replaced with what GR00T outputs
        #print("action used in env: ", action.shape)

        #In the copied env we set the positon ans the extract what we need to pass to our real env
        #copied_env.robots[0].set_robot_joint_positions
        #print("Joint indexes: ", env.robots[0]._ref_joint_pos_indexes)
        #print("Gripper: ", env.robots[0].arms)
        

        all_actions=policy.get_action(full_dict)

        
        #print("Actions len: ", len(all_actions["action.left_arm"]))
        #(7, 7, 6, 6) -> 26
        #all_actions = {k: v[::2] for k, v in all_actions.items()}
        
        for l in range(16):
            #if l % 2 ==0:
            #    continue
            full_number_of_updates+=1
            for r in range(1):
                #action=np.concatenate((all_actions["action.left_arm"][0], 
                #                all_actions["action.right_arm"][0], 
                #                all_actions["action.left_hand"][0],
                #                all_actions["action.right_hand"][0]))
                
                #new_hand_right = obs["robot0_right_gripper_qpos"].copy()
                #new_hand_right[[0, 1, 4, 6, 8, 10]]=all_actions["action.right_hand"][l]

                #new_hand_left = obs["robot0_left_gripper_qpos"].copy()
                #new_hand_left[[0, 1, 4, 6, 8, 10]]=all_actions["action.left_hand"][l] 
            
                #This is now consistent with how the controllers are called
                #action=np.concatenate((all_actions["action.right_arm"][l], 
                #                all_actions["action.left_arm"][l],
                #                states_hands_right,
                #                new_hand_left, 
                #                #all_actions["action.right_hand"][l], #[::-1],
                #                #all_actions["action.left_hand"][l],#[::-1]
                #                )) 
                #print("State: ", states_hands_right[i])
                assert len(all_actions["action.left_arm"]) == 16

                action=np.concatenate((
                                #states_arms_right[i], 
                                all_actions["action.right_arm"][l],#[l],
                                all_actions["action.left_arm"][l], #[l],
                                #states_hands_right[i][::-1],
                                all_actions["action.right_hand"][l][::-1], #l
                                all_actions["action.left_hand"][l][::-1])) #l
                
                all_actions_plot.append(all_actions["action.right_hand"][-1][0])

                arms_only=np.concatenate((all_actions["action.right_arm"][l], all_actions["action.left_arm"][0]))
                #print("Hand Position Predicted: ", all_actions["action.right_arm"][l])#[::-1])
                
                #print(action.shape)

                #print("IDs:", [env.robots[0].sim.model.joint_id2name(joint_id) for joint_id in self.joint_index])

                new_hand_right = obs["robot0_right_gripper_qpos"].copy()
                action_fingers=states_hands_right[0][::-1]#,all_actions["action.right_hand"][i]
                indices = np.array([1, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5])
                new_hand_right=action_fingers[indices]

                new_hand_left = obs["robot0_left_gripper_qpos"].copy()
                new_hand_left[[0, 1, 4, 6, 8, 10]]=all_actions["action.left_hand"][0]
                
                actions=get_axis_angle_actions(obs["robot0_joint_pos"],
                                    obs["robot0_right_gripper_qpos"], 
                                    obs["robot0_left_gripper_qpos"],
                                    arms_only,
                                    new_hand_right,
                                    new_hand_left,
                                    env)

                #action[:-12]=np.concatenate([new_hand_right, new_hand_left])
                #print(action_left_arm)
                #print(action_right_arm)
                #print(action_left_hand)
                #print(action_right_hand)

                #We convert the actions into something the envrionment can use. The env currently uses WholeBodyIK with axis_angles
                #GR00T however outputs joint positions (I think). In he modality.json I can set the rotation type, but since I do not
                #Have a LeRobotSingleDataset that could be really difficult... Im tryng to just get the axis angle input by hand 
                
                #new_left_arm_state=action_left_arm
                #new_right_arm_state=action_right_arm
                #new_left_hand_state=action_left_hand
                #new_right_hand_state=action_right_hand
                
                #env.sim.data.qpos[:] = 

                #OLD
                obs, reward, done, _ = env.step(action)
                print("Done? ", done)
                #if full_number_of_updates % args.skip_frame == 0:
                #frame = env.sim.render(camera_name="robot0_robotview", width=256, height=256)
                #writer.append_data(np.flipud(frame)) 
                
                #OLD
                frame = np.flipud(obs[args.camera + "_image"])#np.flipud(
                writer.append_data(frame)
                print("Saving frame #{}".format(full_number_of_updates))


                #right_hand_indices=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                
                #all_joints_right=["robot0_r_shoulder_pitch", 
                #      "robot0_r_shoulder_roll",
                #      "robot0_r_shoulder_yaw",
                #      "robot0_r_elbow_pitch",
                #      "robot0_r_wrist_yaw",
                #      "robot0_r_wrist_roll",
                #      "robot0_r_wrist_pitch"]
                
                #all_joints_hand_right=[
                #    ['gripper0_right_R_thumb_proximal_yaw_joint', 
                #     'gripper0_right_R_thumb_proximal_pitch_joint', 
                #     'gripper0_right_R_thumb_distal_joint', 
                #     'gripper0_right_R_index_proximal_joint', 
                #     'gripper0_right_R_index_intermediate_joint', 
                #     'gripper0_right_R_middle_proximal_joint', 
                #     'gripper0_right_R_middle_intermediate_joint', 
                #     'gripper0_right_R_ring_proximal_joint',
                #     'gripper0_right_R_ring_intermediate_joint', 
                #     'gripper0_right_R_pinky_proximal_joint', 
                #     'gripper0_right_R_pinky_intermediate_joint']
                #]
                #all_joint_idx_right=[env.sim.model.joint_name2id(el) for el in all_joints_right]
                #right_arm_indices = np.array([env.sim.model.get_joint_qpos_addr(name) for name in all_joints_right])
                #print("Right arm indices: ", right_arm_indices)


                #DIRECT CONTROL
                #print(env.robots[0].print_action_info())

                #This approach is a hybrid approach. the position is set to what we want but then the environment is advanced
                #with a empty, we just do it to advance physics. 
                #print("nq:", env.sim.model.nq)
                #target_qpos=np.zeros(26, )
                #print("Qpos shape: ", target_qpos.shape)
                #target_qpos[right_hand_indices] = new_hand_right
                #target_qpos[right_arm_indices] =  states_arms_right[i]
                
                #robot=env.robots[0]
                
                #robot.sim.data.qpos[right_hand_indices] = new_hand_right
                #robot.sim.data.qpos[right_arm_indices] = states_arms_right[i]
                #env.sim.forward()
                #env.robots[0].set_robot_joint_positions(target_qpos)
                #env.step(action)
                

                #env.sim.data.qpos[right_hand_indices] = new_hand_right
                #env.sim.data.qpos[right_arm_indices] = states_arms_right[i]

                #env.sim.forward()
                #env.sim.render(camera_name="robot0_robotview", width=256, height=256)

            #obs=env._get_observations()

            #Get the next states and the next video frame to input to GR00T
            #obs=env._get_observations()

            
            state_left_arm, state_right_arm, state_left_hand, state_right_hand, video_ego_view=extract_gr00t_data_from_obs(obs, env)
            #joint_order_obs = env.robots[0].robot_model.relevant_joints
            #print("Observation joint order:")
            #for j in joint_order_obs:
            #    print(j)
            #print("Hand State: ", state_right_hand)
            #difference=state_right_hand-states_hands_right[0][::-1] #all_actions["action.right_hand"][l][::-1]
            #difference_right_arm=state_right_arm-states_arms_right[0] #all_actions["action.right_arm"][l]
            #print("Difference Hand: ", difference, "\nDifference Arm: ", difference_right_arm)
            #print("Arm right: ", env.sim.model.joint_name2id("robot0_r_shoulder_pitch"))
            flipped_img = np.flipud(video_ego_view) #flip vertically and horizontally
            imageio.imwrite(f'observation_{i}.png', flipped_img)
            full_dict={"video.ego_view": np.expand_dims(flipped_img, axis=0),
                "state.left_arm": np.array([state_left_arm]), #Somehow if I change these (invert left and right) it works better? 
                "state.right_arm": np.array([state_right_arm]),
                "state.left_hand": np.array([state_left_hand]),#Somehow if I change these (invert left and right) it works better?
                "state.right_hand": np.array([state_right_hand]),
                
                "action.left_arm": all_actions["action.left_arm"],
                "action.right_arm": all_actions["action.right_arm"],
                "action.left_hand": all_actions["action.left_hand"],
                "action.right_hand": all_actions["action.right_hand"],

                "annotation.human.action.task_description": [args.prompt]}
            
            print("Distance gripper and cube: ", env._gripper_to_target(
                gripper=env.robots[0].gripper, target=env.cube.root_body, target_type="body", return_distance=True))
            
            print("\n")
        #if full_number_of_updates == len(states_arms_right):
        #    break
        if done:
            break
    plt.plot(range(120*10), all_actions_plot)
    plt.savefig("plot_action.png")







if __name__=="__main__":
    #text_prompt="Pick up the cube."
    global device
    device="cuda:0" if torch.cuda.is_available() else "cpu" #just to force it to use  a gpu, must be changed depending on what is being used
    print("Device: ", device)


    data_config=DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config=data_config.modality_config()
    modality_transform=data_config.transform()
    EMBODIMENT_TAG="gr1" #should be gr1, but I sometimes is "new_embodiment"
    policy=Gr00tPolicy(
        model_path= "./finetuned-model_trained_20000_no_lora_acc_larger_dataset_new_states_tune_visual_batch_8/checkpoint-20000",#./finetuned-model_trained_20000_no_lora_acc_larger_dataset_new_states_retry/checkpoint-20000",#"./finetuned-model_trained_20000_no_lora_acc_one_vid_same_pos_lr1e-4_wd1e-5/checkpoint-7000",#"./finetuned-model_trained_10000_no_lora_acc_larger_dataset/checkpoint-10000", #"nvidia/GR00T-N1.5-3B", #
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform, 
        device=device)


    parser=argparse.ArgumentParser()

    #Arguments needed for GR00T
    parser.add_argument(
        "--prompt",
        "-p", 
        help="Text prompt passed to the model")

    #Arguments needed for Robosuite
    parser.add_argument(
        "--environment", 
        type=str, 
        default="Lift")

    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="GR1ArmsOnly",
        help="Which robot(s) to use in the env")

    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_robotview",
        help="Camera angle to use for demonstration")
    
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--video_path", type=str, default="video.mp4")
    #parser.add_argument("--renderer",type=str,default="mujoco",help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",)


    args=parser.parse_args()
    create_environment(args)
	#print("Arguments: ", args)


import os
import gr00t
import torch
import tyro
import matplotlib


import matplotlib.pyplot as plt
import numpy as np

from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.utils.peft import get_lora_model
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from transformers import TrainingArguments


device="cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

REPO_PATH=os.path.dirname(os.path.dirname(gr00t.__file__))
DATA_PATH=os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
print("Loading dataset from ...", DATA_PATH)

#modality_configs={
#        "video":ModalityConfig(delta_indices=[0], modality_keys=["video.ego_view"]),
#        "state":ModalityConfig(delta_indices=[0], modality_keys=["state.left_arm", "state.left_hand", "state.left_leg", "state.neck", "state.right_arm", "state.right_hand", "state.right_leg", "state.waist"]),
#        "action":ModalityConfig(delta_indices=[0], modality_keys=["action.left_hand", "action.right_hand"]),
#        "language":ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description", "annotation.human.validity"])}

embodiment_tag=EmbodimentTag.GR1
#dataset=LeRobotSingleDataset(DATA_PATH, modality_configs, video_backend="decord", video_backend_kwargs=None, transforms=None, embodiment_tag=embodiment_tag)

#print("\n"*2)
#print("="*100)
#print(f"{' Humanoid Dataset ':=^100}")
#print("="*100)

#resp=dataset[7]
#any_describe(resp)
#print(resp.keys())

data_config=DATA_CONFIG_MAP["gr1_arms_only"]
modality_config=data_config.modality_config()
modality_transform=data_config.transform()
EMBODIMENT_TAG="gr1"
policy=Gr00tPolicy(
        model_path="nvidia/GR00T-N1-2B", 
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform, 
        device=device)
#print(policy.model)
#print(policy.model.backbone.model.language_model)

modality_config=policy.modality_config
dataset=LeRobotSingleDataset(dataset_path=DATA_PATH, modality_configs=modality_config, video_backend="decord", video_backend_kwargs=None, transforms=None, embodiment_tag=embodiment_tag)

#print(dataset[0])
predicted_action=policy.get_action(dataset[0])
for key, value in predicted_action.items():
    print(key, value.shape)


traj_id=0
max_steps=150
state_joints_across_time=[]
gt_action_joints_across_time=[]


#print("Dataset[0]: ", dataset[0]["video.ego_view"].shape)
#print("Dataset[1]: ", dataset[1]["video.ego_view"].shape)
#print("All zeros? ", dataset[1]["video.ego_view"].mean(axis=-1).astype(np.uint8))

predicted_action=policy.get_action(dataset[0])["action.right_arm"]
print(dataset[0])
all_shapes={el[0]:el[1].shape for el in list(dataset[0].items()) if type(el[1]) != list}
print("All shapes: ", all_shapes)
#print(policy.get_action(dataset[0]))
#print(predicted_action["action.right_arm"])

#for step_count in range(max_steps):
    
    #state_joints=data_point["state.right_arm"][0]
    
    #gt_action_joints= predicted_action.right_arm #data_point["action.right_arm"][0]

    #state_joints_across_time.append(state_joints)
    #gt_action_joints_across_time.append(gt_action_joints)

    #if step_count % (max_steps // sample_images) == 0:
    #    image = data_point["video.ego_view"][0]
    #    images.append(image)

#gt_action_joints_across_time=np.array(gt_action_joints_across_time)

fig, axes=plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))
for i, ax in enumerate(axes):
    ax.plot(predicted_action[:, i], label="predicted action joints")
    ax.legend()
plt.savefig("plot.png") 


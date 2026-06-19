## This file is used to create the rest of what we need for the LeRobot dataset to finetune GR00T

import argparse
import pandas as pd #we use this to read and write parquet files
import h5py
import json

def fix_groot_states(groot_state):
	LEFT_HAND_SLICE = slice(7, 13) 
	RIGHT_HAND_SLICE = slice(29, 35)

	groot_state[LEFT_HAND_SLICE] = groot_state[LEFT_HAND_SLICE][::-1]
	groot_state[RIGHT_HAND_SLICE] = groot_state[RIGHT_HAND_SLICE][::-1]
	return groot_state



def main(args):
	"""The LeRobot datastyle has 3 main parts:
	(1) meta : I already made this separately, it contains info about annotations, the modality config 
	(2) videos:
	(3) data: parquet files that contain the 
	"""
	all_states=[]
	all_actions=[]
	all_annotations_indeces=[]
	all_task_indeces=[]
	all_timestamps=[]
	all_validities=[]
	all_episode_indices=[]
	all_next_rewards=[] #All except the last state have a 0.0 reward, the last one has 1.0
	all_next_dones=[]
	global_episode_index=0
	#global_indeces=[]

	episode_infos=[]
	task=args.task
	global_observations=0
	for f in args.folder:

		#First, I load the data that I have and then loop through each demo, one demo being one episode 
		file=h5py.File(f + "demo.hdf5")["data"]
		#print("Orignal file: ", file)

		#Now we need to reorder the demos
		reordered_file = sorted(file.keys(), key=lambda x: int(x.split("_")[1]))
		number_of_demos=len(reordered_file)
		#print("Reordered file: ", reordered_file)
		whole_sum=0
		local_demo_count=0
		for demo in reordered_file:
			episode_index=str(global_episode_index+local_demo_count)
			print("Local demo count: ", local_demo_count)
			print("Global episode index: ", global_episode_index)
			
			#print("Global epi")

			#episode_index=str(global_episode_index+int(demo.split("_")[1])-1)

			states=[]
			actions=[]
			timestamps=[]
			next_rewards=[]
			next_dones=[]

			task_indeces=[]
			annotations_indeces=[]
			validities=[]
			episode_indices=[]
			global_indeces=[]

			

			print("Length of the episode: ", len(file[demo]["states"]))

			whole_sum+=len(file[demo]["states"])

			for t in range(len(file[demo]["states"])):
				global_observations+=1
				#Iterate through the episode and gather all we need

				#THIS IS BEFORE MY TEST I USED IT FOR ALL MY EXPERIMENTS
				#states.append(file[demo]["groot_states"][t]) #states (I tried to use groot_states now!!)
				#actions.append(file[demo]["groot_states"][t+1]) #collect the groot states, which are basically the next state


				groot_state=file[demo]["groot_states"][t]
				next_groot_state=file[demo]["groot_states"][t+1]

				#TESTING (5. August) I REALIZED THE DATA COLLECTION WRAPPER HAS THE WRONG WAY AROUND FOR HANDS; SO I FIX THIS HERE
				

				states.append(fix_groot_states(groot_state))
				actions.append(fix_groot_states(next_groot_state))

				timestamps.append(file[demo]["timesteps"][t])
				
				#print("How many until end? ", len(file[demo]["states"])- t)

				if (len(file[demo]["states"])- t) != 1:
					next_rewards.append(0.0)
					next_dones.append(False)
				else:
					#print("In end")
					next_rewards.append(1.0)
					next_dones.append(True)

				#Now the episode encompassing values
				task_indeces.append(0) #I think this is the same as annotation index, but Im also not too sure about that
				episode_indices.append(episode_index) #index of the current episode
				global_indeces.append(global_observations) #global observation counter
				annotations_indeces.append(0) #always refer to task 0 (which is pick up the cube)
				validities.append(1) #all tasks are valid although Im not fully sure what that means


			#Once we gathered all the info we need, we save each episode in a parquet file
			new_df=pd.DataFrame(columns=["observation.state", "action", "timestamp", "annotation.human.action.task_description", "task_index", 
										 "annotation.human.validity", "episode_index","index", "next.reward", "next.done"])
			new_df["observation.state"]=states
			new_df["action"]=actions
			new_df["timestamp"]=timestamps
			new_df["annotation.human.action.task_description"]=annotations_indeces
			new_df["task_index"]=task_indeces
			new_df["annotation.human.validity"]=validities
			new_df["episode_index"]=episode_indices
			new_df["index"]=global_indeces
			new_df["next.reward"]=next_rewards
			new_df["next.done"]=next_dones

			
			real_current_index=global_episode_index + local_demo_count#nt(episode_index)
			print(real_current_index)

			episode_name="0"*(6-len(str(real_current_index))) + str(real_current_index)
			print("Episode name: ", episode_name)

			if args.output_folder != None:
				new_df.to_parquet(args.output_folder + f"episode_{episode_name}.parquet")

				if args.episode_json:
					episode_infos.append({"episode_index": global_episode_index + local_demo_count,
										  "task": task,
										  "length": len(file[demo]["states"])})

			local_demo_count+=1
		#print("Number of demos: ", number_of_demos)
		#print("Global episode index: ", global_episode_index)
		global_episode_index+=number_of_demos

	print("Full length: ", global_observations)


	if args.episode_json:

		print("saving episode info at: ", args.output_folder.replace("\\data\\chunk-000\\", "")+"\\meta\\episodes.jsonl")
		with open(args.output_folder.replace("\\data\\chunk-000\\", "")+"\\meta\\episodes.jsonl", "w") as f:
			for item in episode_infos:
				f.write(json.dumps(item) + "\n")
if __name__=="__main__":

	parser=argparse.ArgumentParser()

	parser.add_argument(
		"--folder",
		"-f",
		nargs="+",
		help="Folders from which to get the hdf5 files and the videos")

	parser.add_argument(
		"--output_folder",
		"-o",
		type=str,
		default=None,
		help="Name of the dataset once it is created.")

	parser.add_argument(
		"--episode_json", 
		"-e", 
		action="store_true", 
		help="Whether or not to make a n episodes.json file")
	parser.add_argument(
		"--task", 
		"-t", 
		type=str,
		default="pick up the cube")
	args=parser.parse_args()
	main(args)
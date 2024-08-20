import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pickle
import IPython
import pdb
from PIL import Image
e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        # TODO
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


class Peract2Dataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(Peract2Dataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        #self.camera_names = camera_names
        #todo
        self.camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # dataset_dir: /ailab/group/groups/smartbot/tianyang/Dual_Arm_RLBench/training/bimanual_pick_laptop/all_variations/episodes
        # episodes/episode0/over_shoulder_left_rgb/rgb_0000.png
        # episodes/episode0/low_dim_obs.pkl(variation_description/variation_number)


        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        # TODO
        dataset_path = os.path.join(self.dataset_dir, f'episode{episode_id}')
        #dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
        with open(os.path.join(dataset_path, "low_dim_obs.pkl"), 'rb') as f:
            # qpos_data
            obs = pickle.load(f)
            episode_len = 400
            num_steps = len(obs)
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(num_steps)
            left_joint_positions = np.asarray(obs[start_ts].left.joint_positions, dtype=np.float32)
            left_gripper_positions = np.asarray(obs[start_ts].left.gripper_joint_positions[0], dtype=np.float32)
            left_gripper_positions = np.expand_dims(left_gripper_positions,axis=-1)
            right_joint_positions = np.asarray(obs[start_ts].right.joint_positions, dtype=np.float32)
            right_gripper_positions = np.asarray(obs[start_ts].right.gripper_joint_positions[0], dtype=np.float32)
            right_gripper_positions = np.expand_dims(right_gripper_positions,axis=-1)
            # 可能需要一些预处理，见ee_sim_env.py
            #print(left_joint_positions)
            #print(left_gripper_positions)
            qpos = np.concatenate([right_joint_positions, right_gripper_positions, left_joint_positions, left_gripper_positions], axis=-1)
            #print(qpos[1,:])
            #pdb.set_trace()
            #action
            action = []
            for i in range(num_steps - start_ts):
                left_joint_positions_i = obs[start_ts + i].left.joint_positions
                left_gripper_positions_i = obs[start_ts + i].left.gripper_joint_positions[0]
                left_gripper_positions_i = np.expand_dims(left_gripper_positions_i,axis=-1)
                right_joint_positions_i = obs[start_ts + i].right.joint_positions
                right_gripper_positions_i = obs[start_ts + i].right.gripper_joint_positions[0]
                right_gripper_positions_i = np.expand_dims(right_gripper_positions_i,axis=-1)
                qpo_i = np.concatenate([right_joint_positions_i, right_gripper_positions_i, left_joint_positions_i, left_gripper_positions_i], axis=-1)
                action.append(qpo_i)
            action = np.array(action)
            #image
            image_dict = dict()
            dtype = "rgb"
            for cam_name in self.camera_names:
                camera_full_name = f"{cam_name}_{dtype}"
                data_path = os.path.join(dataset_path, camera_full_name, f"{dtype}_{start_ts:04d}.png")
                image = Image.open(data_path).convert("RGB")
                #pdb.set_trace()
                image_dict[cam_name] = np.array(image)
        # padded_action = 
        action_len = num_steps - start_ts
        padded_action = np.zeros((400,16), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(400)
        is_pad[action_len:] = 1
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images,axis = 0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        #pdb.set_trace()
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

# todo: get_norm_stats_peract2
def get_norm_stats_peract2(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    sample_full_episode = False # hardcode
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode{episode_idx}')
        with open(os.path.join(dataset_path, "low_dim_obs.pkl"), 'rb') as f:
            # qpos_data
            obs = pickle.load(f)
            num_steps = len(obs)
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(num_steps)
            left_joint_positions = np.asarray(obs[start_ts].left.joint_positions, dtype=np.float32)
            left_gripper_positions = np.asarray(obs[start_ts].left.gripper_joint_positions[0], dtype=np.float32)
            left_gripper_positions = np.expand_dims(left_gripper_positions,axis=-1)
            right_joint_positions = np.asarray(obs[start_ts].right.joint_positions, dtype=np.float32)
            right_gripper_positions = np.asarray(obs[start_ts].right.gripper_joint_positions[0], dtype=np.float32)
            right_gripper_positions = np.expand_dims(right_gripper_positions,axis=-1)
            # 可能需要一些预处理，见ee_sim_env.py
            #print(left_joint_positions) #7维度
            #print(left_gripper_positions) #1维度
            qpos = np.concatenate([right_joint_positions, right_gripper_positions, left_joint_positions, left_gripper_positions], axis=-1)
            #action
            '''action = []
            for i in range(num_steps - start_ts):
                left_joint_positions_i = obs[start_ts + i].left.joint_positions
                left_gripper_positions_i = obs[start_ts + i].left.gripper_joint_positions[0]
                left_gripper_positions_i = np.expand_dims(left_gripper_positions_i,axis=-1)
                right_joint_positions_i = obs[start_ts + i].right.joint_positions
                right_gripper_positions_i = obs[start_ts + i].right.gripper_joint_positions[0]
                right_gripper_positions_i = np.expand_dims(right_gripper_positions_i,axis=-1)
                qpo_i = np.concatenate([left_joint_positions_i, left_gripper_positions_i, right_joint_positions_i, right_gripper_positions_i], axis=-1)
                action.append(qpo_i)
            action = np.array(action)'''
        all_qpos_data.append(torch.from_numpy(qpos))
        #all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    #all_action_data = torch.stack(all_action_data)
    #all_action_data = all_action_data

    # normalize action data
    #action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    #action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    #action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    #pdb.set_trace()
    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data_peract2(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_peract2(dataset_dir, num_episodes)
    #pdb.set_trace()
    # construct dataset and dataloader
    train_dataset = Peract2Dataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = Peract2Dataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

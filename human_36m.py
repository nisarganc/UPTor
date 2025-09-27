"""Human36M Dataset.

The data is preprocessed and sampled just as in [3]. 
The dataloader has been adapted from [7] and [8].
Evaluation protocol is adopted from [8].

See:
[1] https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
[2] https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
[3] https://arxiv.org/abs/1705.02445
[4] https://github.com/una-dinosauria/human-motion-prediction
[5] https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
[6] https://github.com/eth-ait/spl/blob/master/preprocessing/preprocess_h36m.py
[7] https://arxiv.org/pdf/2109.07531.pdf
[8] https://github.com/mmahdavian/STPOTR/blob/main/data/H36MDataset_v3.py

"""

import numpy as np
import os
import argparse
import sys
import yaml
import math
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import utils.utils as utils

_MIN_STD = 1e-4 #0.0001
_MAJOR_JOINTS = np.array([-1, 0, 1, 2, 5, 6, 7, 11, 12, 13, 14, 16, 17, 18, 24, 25, 26])+1
_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
window = 2

######################### Helper Functions ##############################

def collate_fn(batch):
  """Collate function for data loaders."""
  
  e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
  d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))  
  src_mask = torch.from_numpy(np.stack([e['src_mask'] for e in batch]))  

  d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
  org = torch.from_numpy(np.stack([e['original_data']for e in batch]))  
  trans = torch.from_numpy(np.stack([e['translation']for e in batch]))
  rot = torch.from_numpy(np.stack([e['rotation']for e in batch]))
  batch_ = {
      'encoder_inputs': e_inp,
      'src_mask': src_mask,
      'decoder_inputs': d_inp,

      'original_data': org,
      'decoder_outputs': d_out,
      'translation': trans,
      'rotation': rot
  }
  return batch_

def collate_fn2(batch):
    """Collate function for data loaders."""
    sizes = [e['encoder_inputs'].shape[0] for e in batch]
    e_inp = torch.from_numpy(np.concatenate([e['encoder_inputs'] for e in batch]))
    d_inp = torch.from_numpy(np.concatenate([e['decoder_inputs'] for e in batch]))
    src_mask = torch.from_numpy(np.concatenate([e['src_mask'] for e in batch])) 


    d_out = torch.from_numpy(np.concatenate([e['decoder_outputs'] for e in batch]))
    org = torch.from_numpy(np.concatenate([e['original_data']for e in batch]))  
    trans = torch.from_numpy(np.concatenate([e['translation']for e in batch]))
    rot = torch.from_numpy(np.concatenate([e['rotation']for e in batch]))

    batch_ = {
        'encoder_inputs': e_inp,
        'src_mask': src_mask,
        'decoder_inputs': d_inp,

        'original_data': org,
        'decoder_outputs': d_out,
        'translation': trans,
        'rotation': rot
    }
    return batch_

def _find_indices_srnn(data, action):
  """Find the same action indices as in SRNN. See [1]."""
  # Used a fixed dummy seed, following [5]
  SEED = 1234567890
  rng = np.random.RandomState(SEED)

  subject1 = 'S9'
  subject2 = 'S11'

  T1 = data[(subject1, action)].shape[0]
  T2 = data[(subject2, action)].shape[0]
  prefix, suffix = 50, 100

  idx = []
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  idx.append(rng.randint(16,T1-prefix-suffix))
  idx.append(rng.randint(16,T2-prefix-suffix))
  return idx 

########################## Human 3.6M Dataset #######################

class Human36MDataset(torch.utils.data.Dataset):
  def __init__(self, 
              params=None,
              mode='train', 
              action="all",
              **kwargs):
    super(Human36MDataset, self).__init__(**kwargs)
    self._actions = action
    self._params = params
    self._data = {}
    self._mode = mode
    self._test_n_seeds = params['eval_num_seeds']
    print('[INFO] (Human3.6MDataset) mode: {}'.format(self._mode))
    self._framerates = {} # frq: stepsize
    self._selectedfrq = self._params['selected_frequency']
    self.load_data() 
    self._data_keys = self._data.keys()
    
  def collectfrq(self, org_frq):
    min_frq = 4
    max_frq = 50
    for divisor in range(1, org_frq + 1):
        gcd = math.gcd(org_frq, divisor)
        frequency = int(org_frq // gcd)
        if frequency >= min_frq and frequency <= max_frq and frequency not in self._framerates.keys():
            self._framerates[frequency] = int(org_frq // frequency)  

  def preprocess_sequence(self, action_sequence):
    """Selection the good joints and convert to required format.
    Args:
      action_sequence: [n_frames, 96]
    """
    total_frames, D = action_sequence.shape
    # total_framesx32x3
    data_sel = action_sequence.reshape((total_frames, -1, 3))
    # total_framesx21x3
    data_sel = data_sel[:, _MAJOR_JOINTS]
    # total_frames x n_joints*dim_per_joint
    data_sel = data_sel.reshape((total_frames, -1))

    return data_sel

  def load_data(self):
    """Loads all CMU dataset into memory."""
    self._data = {} # {(11): np.array}

    # Load the YAML file
    with open('human_36m/human_36m.yaml', 'r') as yaml_file:
        dataset_attributes = yaml.safe_load(yaml_file)

    # Access the variables
    self._num_joints = dataset_attributes['_NMAJOR_JOINTS']
    self._pose_dim = 3*self._num_joints
    self._bone_joint1_idx = dataset_attributes['bone_joint1_idx']
    self._bone_joint2_idx = dataset_attributes['bone_joint2_idx']
    self._joint_names = dataset_attributes['joint_names']
    self._frq = dataset_attributes['frq']
    self._red_src = dataset_attributes['source_seq_len'] #25 = 0.5s
    self._red_tgt = dataset_attributes['target_seq_len'] #100 = 2s   

    self.collectfrq(self._frq)
    self._source_seq_len = 5 #int(self._red_src/self._framerates[self._selectedfrq]) #5
    self._target_seq_len = 20 #int(self._red_tgt/self._framerates[self._selectedfrq]) #20

    all_dataset = []
    self.data_sequence_idx = []
    data_address = 'human_36m/data.npz'
    my_data = np.load(data_address, allow_pickle=True)['positions_3d'].item()

    for subject, actions in my_data.items():
      
      if self._mode == 'train':
        if subject == 'S9' or subject == 'S11':
          continue
      else:
        if subject == 'S1' or subject == 'S5' or subject == 'S6' or subject == 'S7' or subject == 'S8':
          continue
    
      for action, action_sequence in actions.items():
        if action == self._actions or self._actions == "all": 
          entry_key = subject, action
          n_frames, joints, dim = action_sequence.shape  

          sequence = self.preprocess_sequence(action_sequence.reshape(-1,joints*dim))

          self._data[entry_key] = {}
          self._data[entry_key] = sequence
          frames, dim = self._data[entry_key].shape

          all_dataset.append(self._data[entry_key])

          valid_frames = np.arange(0, frames - self._red_src - self._red_tgt + 1, 1)
          self.data_sequence_idx.extend(zip([entry_key] * len(valid_frames), valid_frames.tolist()))  

    all_dataset = np.concatenate(all_dataset, axis=0)
    print('[INFO] ({}) Dataset size: {}'.format(self.__class__.__name__, all_dataset.shape))

    # suffle self.data_sequence_idx items
    np.random.shuffle(self.data_sequence_idx)
    np.random.shuffle(self.data_sequence_idx)

  ########################### other helper functions ##################################
  def __len__(self):
    return len(self.data_sequence_idx) 

  def get_pose_dim(self):
    """Returns the pose dimension as a flattened vector."""
    return self._pose_dim
  
  ########################### get data item ############################################
  
  def apply_random_rotation(self, data_sel, max_angle):
    '''sequence should have shape [T, 90]'''
    angle =  np.random.uniform(-max_angle, max_angle)

    # Create a rotation matrix around the z-axis
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    T, _ = data_sel.shape
    data_sel=data_sel.reshape(T, self._num_joints, 3) 
    rotated_sequence = np.zeros_like(data_sel)

    # Generate subsequent skeleton frames by adding the distances
    for t in range(T):
        rotated_sequence[t] = np.dot(rotation_matrix, data_sel[t].transpose()).transpose()    

    return rotated_sequence.reshape(T, self._num_joints*3)
  
  def apply_random_translation(self, data_sel, max_translation):
    random_number = np.random.randint(0, max_translation)
    add_tensor = np.array([random_number, random_number, random_number])
    
    T, _ = data_sel.shape
    data_sel=data_sel.reshape(-1, self._num_joints, 3) 
    translated_sequence = np.zeros_like(data_sel)

    translation = np.tile(add_tensor, (self._num_joints, 1)) # 30*3

    for t in range(T):
      translated_sequence[t] = data_sel[t] + translation

    return translated_sequence.reshape(T, self._num_joints*3)

  def traslate_to_origin(self, sequence, source_seq_len):
    T, _ = sequence.shape # seq shape [T, 90]
    sequence=sequence.reshape(-1, self._num_joints, 3) # reshape to [T, 30, 3]
    #translated_sequence = np.zeros_like(sequence)

    # Compute the translation vector
    output_frame = sequence[source_seq_len] # 30*3
    translation = -output_frame[0]#pelvis 3
    trans = translation.copy()

    translation = np.tile(translation, (self._num_joints, 1)) # 30*3
    # Translate the entire sequence
    translated_sequence = sequence + translation
    return translated_sequence, -trans

  def rotate_to_posxaxis(self, data_sel, source_seq_len):
    '''sequence should have shape [T, 90]'''
    T, _, _ = data_sel.shape
    data_sel=data_sel.reshape(-1, self._num_joints, 3) 
    rotated_sequence = np.zeros_like(data_sel)

    # Compute the angle between motion direction and positive x-axis
    delta_x = data_sel[source_seq_len, 0, 0] - data_sel[source_seq_len-window, 0, 0]
    delta_y = data_sel[source_seq_len, 0, 1] - data_sel[source_seq_len-window, 0, 1]

    # Check for negligible movement, with a threshold (e.g., 1e-5)
    if abs(delta_x) < 1e-2 and abs(delta_y) < 1e-2:
      angle = 0.0
    else:
      angle = np.arctan2(delta_y, delta_x)

    # Create a rotation matrix around the z-axis and rotate
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle), 0],
                                [np.sin(-angle), np.cos(-angle), 0],
                                [0, 0, 1]])
    for t in range(T):
        rotated_sequence[t] = np.dot(rotation_matrix, data_sel[t].T).T   
    return rotated_sequence.reshape(T, self._num_joints*3), angle

  def __getitem__ (self, index):
    if self._mode == 'train':
      return self._get_item_train(index)
    else:
      return self._get_item_eval_total()

  def _get_item_train(self, index):
    """Get item for the training mode."""
    pose_size = self._pose_dim
    source_seq_len = self._source_seq_len
    target_seq_len = self._target_seq_len  
    
    idx, start_frame = self.data_sequence_idx[index]

    data_sel_full = self._data[idx][start_frame:(start_frame + self._red_src + self._red_tgt), :]
    data_sel = data_sel_full[::5]
    T, _ = data_sel.shape      
    
    original_data = np.zeros((self._source_seq_len+self._target_seq_len, pose_size), 
                             dtype=np.float32)
    original_data[:source_seq_len+target_seq_len, :] = data_sel.copy()

    # Apply random translation and rotation 
    if self._params['random_transrot']:
      data_sel = self.apply_random_rotation(data_sel, 3.14)
      data_sel = self.apply_random_translation(data_sel, 10)

    # Transform to origin before training
    translation = np.zeros(3)
    rotation = np.zeros(1)
    if self._params['transform_torigin']:
      data_sel, translation = self.traslate_to_origin(data_sel, source_seq_len-1)
      data_sel, rotation = self.rotate_to_posxaxis(data_sel, source_seq_len-1)

    # INITIALIZE
    encoder_inputs = np.zeros((self._source_seq_len, pose_size), dtype=np.float32)
    decoder_inputs = np.zeros((self._target_seq_len, pose_size), dtype=np.float32)
    decoder_outputs = np.zeros((self._target_seq_len, pose_size), dtype=np.float32)
    src_mask = np.zeros((self._source_seq_len), dtype=np.float32)

    encoder_inputs[:source_seq_len, :] = data_sel[:source_seq_len, :]
    decoder_inputs[:target_seq_len, :] = np.repeat(encoder_inputs[-1:, :], 
                                                   target_seq_len, axis=0)
    decoder_outputs = data_sel[source_seq_len:, 0:pose_size]

    return {
        'encoder_inputs': encoder_inputs,
        'src_mask': src_mask,
        'decoder_inputs': decoder_inputs,
        
        'original_data': original_data,
        'decoder_outputs': decoder_outputs,
        'translation': translation,
        'rotation': rotation    
        }
  
  # Evaluation protocol from STPOTR
  def _get_item_eval_total(self):
    pose_size = self._pose_dim
    source_seq_len = self._source_seq_len
    target_seq_len = self._target_seq_len   

    batch_ = [self._sample_batch_eval(source_seq_len, target_seq_len, 
                                      pose_size, sub, action) for sub, action in self._data.keys()]
    batch = collate_fn2(batch_)

    encoder_inputs = batch['encoder_inputs'].view(-1, source_seq_len, pose_size)
    decoder_inputs = batch['decoder_inputs'].view(-1, target_seq_len, pose_size)
    original_data = batch['original_data'].view(-1, source_seq_len+target_seq_len, 
                                                pose_size)
    decoder_outputs = batch['decoder_outputs'].view(-1, target_seq_len, pose_size)

    src_mask = batch['src_mask'].view(-1, source_seq_len)
    translation = batch['translation'].view(-1, 3)
    rotation = batch['rotation']


    return {
    'encoder_inputs': encoder_inputs,
    'src_mask': src_mask,
    'decoder_inputs': decoder_inputs,
    
    'original_data': original_data,
    'decoder_outputs': decoder_outputs,
    'translation': translation,
    'rotation': rotation    
    }

  def _sample_batch_eval(self, source_seq_len, target_seq_len, 
                         pose_size, sub, action):
      
    data_action_full = self._data[sub, action]  
    data_select = data_action_full[::5]
    tot_seq, _ = data_select.shape
    seq_len = source_seq_len+target_seq_len 
    seq1 = tot_seq // seq_len   

    encoder_inputs = np.zeros((seq1, self._source_seq_len, pose_size),
                              dtype=np.float32)
    decoder_inputs = np.zeros((seq1, self._target_seq_len, pose_size), 
                              dtype=np.float32)
    src_mask = np.zeros((seq1, self._source_seq_len), dtype=np.float32)

    original_data = np.zeros((seq1, self._source_seq_len+self._target_seq_len, pose_size), 
                             dtype=np.float32)
    decoder_outputs = np.zeros((seq1, self._target_seq_len, pose_size), 
                               dtype=np.float32)
    translation = np.zeros((seq1, 3), dtype=np.float32)
    rotation = np.zeros((seq1), dtype=np.float32)

    src_mask = np.zeros((seq1, self._source_seq_len), 
                        dtype=np.float32)

    for i in range(seq1):
      data_sel = data_select[i*(seq_len):(i+1)*(seq_len), :]
      original_data[i, :source_seq_len+target_seq_len, :] = data_sel.copy()
      if self._params['transform_torigin']:
        data_sel, translation[i, :] = self.traslate_to_origin(data_sel, 
                                                              source_seq_len-1)
        data_sel, rotation[i] = self.rotate_to_posxaxis(data_sel, 
                                                        source_seq_len-1)

      encoder_inputs[i, :source_seq_len, :] = data_sel[:source_seq_len, :]
      decoder_inputs[i, :target_seq_len, :] = np.repeat(encoder_inputs[i, -1:, :], 
                                                        target_seq_len, axis=0)
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, 0:pose_size]

    return {
        'encoder_inputs': encoder_inputs,
        'src_mask': src_mask,
        'decoder_inputs': decoder_inputs,
        
        'original_data': original_data,
        'decoder_outputs': decoder_outputs,
        'translation': translation,
        'rotation': rotation    
        }

  
  ######################### data post processing ########################################
  def rotation_matrix(self, angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])

  def untransform_data(self, seq, translation, angle):
    # sequence shape [Batch_size, Time_Sequence, joints, pose_dim]
    if translation is None and angle is None:
      return seq
    
    B, T, _, _ = seq.shape 
    sequence = seq.numpy()
    untransformed_sequence = sequence.copy()
    
    for b in range(B):
      for t in range(T):
          untransformed_sequence[b, t] = np.dot(self.rotation_matrix(angle[b]), 
                                                untransformed_sequence[b, t].T).T   

    translation = translation.numpy()[:, np.newaxis, np.newaxis, :] # B*3 -> B*1*1*3
    translation = np.repeat(translation, T, axis=1) # B*1*1*3 -> B*T*1*3
    translation = np.repeat(translation, self._num_joints, axis=2) # B*T*1*3 -> B*T*J*3

    untransformed_sequence = np.add(untransformed_sequence, translation)

    return untransformed_sequence
      
  def plot_predictions(self, motion_input, motion_target, 
                       motion_pred, nb_iter=None, path=None):  

    motion_input = motion_input.reshape(-1, self._source_seq_len, self._num_joints, 3)
    motion_target = motion_target.reshape(-1, self._target_seq_len, self._num_joints, 3)
    motion_pred = motion_pred.reshape(-1, self._target_seq_len, self._num_joints, 3)

    batch, frames_input, joints, _ = motion_input.shape
    _, frames_output, _, _ = motion_target.shape

    item = np.random.randint(0, batch)
    motion_input_= motion_input[item,:,:,:]
    motion_target_= motion_target[item,:,:,:]
    motion_pred_= motion_pred[item,:,:,:]

    fig = plt.figure(figsize=(6, 6))
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-4 , -4 )
    ax.set_ylim(-4 , -4 )
    ax.set_zlim(0 , 2 )   
    ax.set_xlabel('X', labelpad=40)
    ax.set_ylabel('Y', labelpad=40)
    ax.set_zlabel('Z')

    ax.set_xticks(np.arange(-4, 4, 1)) 
    ax.set_yticks(np.arange(-4, 4, 1))  
    ax.set_zticks(np.arange( 0, 2, 5)) 

    lines = []
    scatter = None
    frames =[]
    for i in range(frames_input):
      
      joints = motion_input_[i,:,:] 
          
      # Plot the skeleton joints
      scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                           c='black', marker='o', s=2)
    
      # Connect the joints with lines to form the skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj = ax.plot([joints[joint1][0], joints[joint2][0]], 
                           [joints[joint1][1], joints[joint2][1]], 
                           [joints[joint1][2], joints[joint2][2]], 
                            color='g', lw=2)
        lines.extend(line_obj)

      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)

      plt.pause(0.1)  
      
      # Clear the lines and scatter from previous frame
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()

    scatter = None
    scatter2 = None
    for j in range(frames_output):
      target = motion_target_[j,:,:]
      pred = motion_pred_[j,:,:]

      # Plot the skeleton joints
      scatter = ax.scatter(target[:, 0], target[:, 1], target[:, 2], 
                           c='black', marker='o', s=2)
      scatter2 = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], 
                            c='black', marker='o', s=2)
      
      # Connect the joints with lines to form the target skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj1 = ax.plot([target[joint1][0], target[joint2][0]], 
                            [target[joint1][1], target[joint2][1]], 
                            [target[joint1][2], target[joint2][2]], 
                            color='g', lw=2)
        lines.extend(line_obj1)
        line_obj2 = ax.plot([pred[joint1][0], pred[joint2][0]], 
                            [pred[joint1][1], pred[joint2][1]], 
                            [pred[joint1][2], pred[joint2][2]], 
                            color='red', lw=2)
        lines.extend(line_obj2) 
      
      # Trajectory
      if j > 0:
        ax.plot([motion_target_[j,0,0], motion_target_[j-1,0,0]], 
                [motion_target_[j,0,1], motion_target_[j-1,0,1]], 
                [motion_target_[j,0,2], motion_target_[j-1,0,2]], 
                color='g', lw=2)        
        ax.plot([motion_pred_[j,0,0], motion_pred_[j-1,0,0]], 
                [motion_pred_[j,0,1], motion_pred_[j-1,0,1]], 
                [motion_pred_[j,0,2], motion_pred_[j-1,0,2]], 
                color='red', lw=2)
            
      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)
      plt.pause(0.1)  
      
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()
      if scatter2:
        scatter2.remove()     

    clip = ImageSequenceClip(frames, fps=6)
    if not os.path.exists(f'{path}/vis'):
      os.makedirs(f'{path}/vis')
    clip.write_videofile(f'{path}/vis/batch_{nb_iter:04d}_item_{item:04d}.mp4', fps=6)

    plt.close(fig)       

  def plot_predictions_frames(self, motion_input, motion_target, 
                       motion_pred, nb_iter=None, path=None):  

    motion_input = motion_input.reshape(-1, self._source_seq_len, self._num_joints, 3)
    motion_target = motion_target.reshape(-1, self._target_seq_len, self._num_joints, 3)
    motion_pred = motion_pred.reshape(-1, self._target_seq_len, self._num_joints, 3)

    batch, frames_input, joints, _ = motion_input.shape
    _, frames_output, _, _ = motion_target.shape

    item = np.random.randint(0, batch)
    motion_input_= motion_input[item,:,:,:]
    motion_target_= motion_target[item,:,:,:]
    motion_pred_= motion_pred[item,:,:,:]

    fig = plt.figure(figsize=(6, 6))
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    
    plt.ion()
    ax.set_xlim(-2 , 2 )
    ax.set_ylim(-2 , 2 )
    ax.set_zlim( -1 , 2 ) 
    ax.set_xticks(np.arange(-2, 2, 0.5))   
    ax.set_yticks(np.arange(-2, 2, 0.5))  
    ax.set_zticks(np.arange(-1, 2, 5))  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.xaxis.pane.set_edgecolor('1')  # set edge color to light gray
    ax.yaxis.pane.set_edgecolor('1')
    ax.zaxis.pane.set_edgecolor('1')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo["grid"].update({"color": "1", "linewidth": 0, "linestyle": "--"})
    ax.yaxis._axinfo["grid"].update({"color": "1", "linewidth": 0, "linestyle": "--"})
    ax.zaxis._axinfo["grid"].update({"color": "1", "linewidth": 0, "linestyle": "--"})

    lines = []
    scatter = None
    frames =[]
    for i in range(frames_input):
      
      joints = motion_input_[i,:,:] 
          
      # Plot the skeleton joints
      scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                           c='g', marker='o', s=3)  
    
      # Connect the joints with lines to form the skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj = ax.plot([joints[joint1][0], joints[joint2][0]], 
                           [joints[joint1][1], joints[joint2][1]], 
                           [joints[joint1][2], joints[joint2][2]], 
                            color='g', lw=2)
        lines.extend(line_obj)

      # # print joint names
      # for j, txt in enumerate(self._joint_names):
      #   ax.text(joints[j, 0], joints[j, 1], joints[j, 2], '%s' % (txt), size=10, color='k')

      # plt.pause(10)  
      # plt.savefig('huvman36m_joints.png')  
      # exit() 

      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)

      plt.pause(0.1)  
      
      # Clear the lines and scatter from previous frame
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()
    plt.pause(5) 
    scatter = None
    scatter2 = None
    for j in range(frames_output):
      alpha_val = j/(frames_output+10)
      # subtract all joints from root joint

      target = motion_target_[j,:,:]
      pred = motion_pred_[j,:,:]
      if j == frames_output - 1:
          ax.view_init(elev=90, azim=-90) 

      # Plot the skeleton joints
      scatter = ax.scatter(target[:, 0], target[:, 1], target[:, 2], 
                           c='darkgreen', marker='o', s=5)
      scatter2 = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], 
                            c='darkred', marker='o', s=5)
      
      # Connect the joints with lines to form the target skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj1 = ax.plot([target[joint1][0], target[joint2][0]], 
                            [target[joint1][1], target[joint2][1]], 
                            [target[joint1][2], target[joint2][2]], 
                            color='g', alpha=0.9, lw=2.8)
        lines.extend(line_obj1)
        line_obj2 = ax.plot([pred[joint1][0], pred[joint2][0]], 
                            [pred[joint1][1], pred[joint2][1]], 
                            [pred[joint1][2], pred[joint2][2]], 
                            color='red', alpha=0.9, lw=2.8)
        lines.extend(line_obj2) 
            
      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)
      plt.pause(0.8)  
      if nb_iter is not None and path is not None and j % 2 == 0:
        if not os.path.exists(f'{path}/vis'):
          os.makedirs(f'{path}/vis')  
        if not os.path.exists(f'{path}/vis/Eating'):
          os.makedirs(f'{path}/vis/Eating')
        plt.savefig(f'{path}/vis/Eating/pose_output{j:04d}.png')
   
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()
      if scatter2:
        scatter2.remove() 
    ax.set_xlim(-2 , 2 )
    ax.set_ylim(-2 , 2 )        
    for j in range(frames_output):    
      # Trajectory 
      if j > 0:
        ax.plot([motion_target_[j,0,0], motion_target_[j-1,0,0]], 
                [motion_target_[j,0,1], motion_target_[j-1,0,1]], 
                [motion_target_[j,0,2], motion_target_[j-1,0,2]], 
                color='g', alpha=alpha_val, lw=2.5)        
        ax.plot([motion_pred_[j,0,0], motion_pred_[j-1,0,0]], 
                [motion_pred_[j,0,1], motion_pred_[j-1,0,1]], 
                [motion_pred_[j,0,2], motion_pred_[j-1,0,2]], 
                color='red', alpha=alpha_val, lw=2.5)
    ax.xaxis._axinfo["grid"].update({"color": "0.8", "linewidth": 1, "linestyle": "--"})
    ax.yaxis._axinfo["grid"].update({"color": "0.8", "linewidth": 1, "linestyle": "--"})
    ax.zaxis._axinfo["grid"].update({"color": "0.8", "linewidth": 1, "linestyle": "--"})
    plt.savefig(f'{path}/vis/Eating/traj_output{j:04d}.png')

    # clip = ImageSequenceClip(frames, fps=6)
    # if not os.path.exists(f'{path}/vis'):
    #   os.makedirs(f'{path}/vis')
    # clip.write_videofile(f'{path}/vis/batch_{nb_iter:04d}_item_{item:04d}.mp4', fps=6)

    plt.close(fig) 
    exit()

  ################################ extra debugging functions ################################
  def aug_test(self, index):
    source_seq_len = self._source_seq_len
    target_seq_len = self._target_seq_len

    idx, start_frame = self.data_sequence_idx[index]

    data_sel_full = self._data[idx][start_frame:(start_frame + self._red_src + self._red_tgt), :]
    data_sel = data_sel_full[::self._framerates[self._selectedfrq]]

    data_sel_aug, trans = self.traslate_to_origin(data_sel, source_seq_len-1)
    data_sel_aug, rot = self.rotate_to_posxaxis(data_sel_aug, source_seq_len-1)
    # print(rot)
    # data_sel_aug = self.apply_random_rotation(data_sel, 3.14)

    return data_sel, data_sel_aug   
 
  # Plotting
  def plot_augumentation(self, motion_input, motion_input_augmented):  

    motion_target_ = motion_input.reshape(-1, self._num_joints, 3)
    motion_pred_ = motion_input_augmented.reshape(-1, self._num_joints, 3)
    frames_input, _, _ = motion_pred_.shape

    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    
    plt.ion()

    ax.set_xlim(-2 , 2 )
    ax.set_ylim(-2 , 2 )
    ax.set_zlim(-2 , 2 )  
    ax.set_xlabel('X', labelpad=40)
    ax.set_ylabel('Y', labelpad=40)
    ax.set_zlabel('Z')

    lines = []
    frames =[]
    scatter = None
    scatter2= None
    for j in range(frames_input):
      target = motion_target_[j,:,:]
      pred = motion_pred_[j,:,:]

      # Plot the skeleton joints
      scatter = ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='black', marker='o', s=2)
      scatter2 = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='black', marker='o', s=2)
      
      # Connect the joints with lines to form the target skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj1 = ax.plot([target[joint1][0], target[joint2][0]], 
                            [target[joint1][1], target[joint2][1]], 
                            [target[joint1][2], target[joint2][2]], 
                            color='g', lw=2)
        lines.extend(line_obj1)

      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        colorsel = 'r' if j < self._source_seq_len else 'b'
        line_obj2 = ax.plot([pred[joint1][0], pred[joint2][0]], 
                            [pred[joint1][1], pred[joint2][1]], 
                            [pred[joint1][2], pred[joint2][2]], 
                            color=colorsel, lw=2)
        lines.extend(line_obj2)
      
      if j > 0:
        ax.plot([motion_target_[j,0,0], motion_target_[j-1,0,0]], 
                [motion_target_[j,0,1], motion_target_[j-1,0,1]], 
                [motion_target_[j,0,2], motion_target_[j-1,0,2]], 
                color='g', alpha=0.1+(0.1*i/10), lw=2)
        
      
      if j > 0:
        ax.plot([motion_pred_[j,0,0], motion_pred_[j-1,0,0]], 
                [motion_pred_[j,0,1], motion_pred_[j-1,0,1]], 
                [motion_pred_[j,0,2], motion_pred_[j-1,0,2]], 
                color='r', alpha=0.1+(0.1*i/10), lw=2)      
            
      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)
      plt.pause(0.1)  
      
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()
      if scatter2:
        scatter2.remove()    
    
    # Create a mp4 from the frames
    # clip = ImageSequenceClip(frames, fps=6)
    # clip.write_videofile(f'dataset_tranform.mp4', fps=7)
    
    plt.close(fig)

  def plot_dataset(self, motion_input):  
    motion_input = motion_input.reshape(-1, self._num_joints, 3)
    frames_input, joints, _ = motion_input.shape

    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')
    
    plt.ion()

    ax.set_xlim(-2 , 2 )
    ax.set_ylim(-2 , 2)
    ax.set_zlim(-2 , 2 )  
    ax.set_xlabel('X', labelpad=40)
    ax.set_ylabel('Y', labelpad=40)
    ax.set_zlabel('Z')

    lines = []
    scatter = None
    frames =[]

    # Plot each frame of the input skeleton
    for i in range(frames_input):
      joints = motion_input[i,:,:] 
          
      # Plot the skeleton joints
      scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                           c='black', marker='o', s=2)
    
      # Connect the joints with lines to form the skeleton
      for edge in zip(self._bone_joint1_idx, self._bone_joint2_idx):
        joint1, joint2 = edge
        line_obj = ax.plot([joints[joint1][0], joints[joint2][0]], 
                           [joints[joint1][1], joints[joint2][1]], 
                           [joints[joint1][2], joints[joint2][2]], 
                            color='g', lw=2)
        lines.extend(line_obj)
      
      if i > 0:
        ax.plot([motion_input[i,0,0], motion_input[i-1,0,0]], 
                [motion_input[i,0,1], motion_input[i-1,0,1]], 
                [motion_input[i,0,2], motion_input[i-1,0,2]], 
                color='g', lw=2)
        
      frame = np.array(fig.canvas.renderer._renderer)
      frames.append(frame)

      plt.pause(0.1)   
      for line in lines:
        line.remove()
      lines.clear()
      if scatter:
        scatter.remove()

    # # Create a mp4 from the frames
    # clip = ImageSequenceClip(frames, fps=6)
    # clip.write_videofile(f'dataset_sequence.mp4', fps=7)

    plt.close(fig)

############################# Dataset Factory #######################################

def dataset_factory(params):
  """Defines the datasets that will be used for training and validation."""

  train_dataset = Human36MDataset(params, mode='train')
  train_dataset_fn = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=params['batch_size'],
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
  )

  eval_dataset = Human36MDataset(params, mode='eval', action="all")
  eval_dataset_fn = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    #collate_fn=collate_fn    
  ) 

  return train_dataset_fn, eval_dataset_fn

#################################### Main Function ###############################

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', 
                      type=int, default=10)
  parser.add_argument('--eval_num_seeds', 
                      type=int, default=8)
  parser.add_argument('--transform_torigin', 
                      action='store_true', default=True)
  parser.add_argument('--random_transrot', 
                      action='store_true', default=False)
  parser.add_argument('--selected_frequency', 
                      type=int, default=10)
  parser.add_argument('--actions', 
                      type=str, default="all")

  args = parser.parse_args()
  params = vars(args)

  dataset_t, dataset_e = dataset_factory(params)
  print('train loader size: {}'.format(len(dataset_t)))
  print('eval loader size: {}'.format(len(dataset_e)))
  sample = next(iter(dataset_e))
  print(sample['encoder_inputs'].shape)

  data_class = Human36MDataset(params, mode='train')
  for current_step, data in enumerate(dataset_t):
    i=9
    print(data['encoder_inputs'][i].shape)
    print(data['decoder_inputs'][i].shape)
    print(data['decoder_outputs'][i].shape)
    print(data['original_data'][i].shape)
    print(data['src_mask'][i])

    getdata= data['encoder_inputs'][i]
    getdatainp = data['decoder_inputs'][i]    
    getdataout = data['decoder_outputs'][i]
    getdataorg = data['original_data'][i]
    break
 
  test, test_aug = data_class.aug_test(70)
  print(test.shape, test_aug.shape)

  # data_class.plot_augumentation(test, test_aug)
  # dataset_t.dataset.plot_predictions_frames(getdata, getdatainp, 
  #                             getdataout, nb_iter=0, path='test')
  # data_class.plot_dataset(getdataorg)
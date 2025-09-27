"""CMU-Mocap Walking data-subset

The data is preprocessed and sampled just as in [3]. 
The dataloader has been developed as in [7] and [8].

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

import os
import glob
import argparse
import sys
import yaml
import math

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

_MAJOR_JOINTS = np.array([
  0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 25])
joint_edges_major_new = [
  [0, 1],   # maps to [0, 1]
  [0, 5],   # maps to [0, 6]
  [0, 9],   # maps to [0, 11]
  [1, 2],   # maps to [1, 2]
  [2, 3],   # maps to [2, 3]
  [3, 4],   # maps to [3, 4]
  [5, 6],   # maps to [6, 7]
  [6, 7],   # maps to [7, 8]
  [7, 8],   # maps to [8, 9]
  [9, 10],  # maps to [11, 12]
  [10, 11], # maps to [12, 13]
  [11, 12], # maps to [13, 14]
  [12, 13], # maps to [14, 15]
  [13, 14], # maps to [15, 16]
  [11, 15], # maps to [13, 18]
  [11, 16]  # maps to [13, 25]
]

######################### Helper Functions ##############################

def collate_fn(batch):
  """Collate function for data loaders."""

  e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
  time_src = torch.from_numpy(np.stack([e['time_stamp_src']for e in batch]))
  src_mask = torch.from_numpy(np.stack([e['src_mask']for e in batch]))
  d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))  
  time_tgt = torch.from_numpy(np.stack([e['time_stamp_tgt']for e in batch]))

  org = torch.from_numpy(np.stack([e['original_data']for e in batch]))  
  d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
  trans = torch.from_numpy(np.stack([e['translation']for e in batch]))
  rot = torch.from_numpy(np.stack([e['rotation']for e in batch]))
  batch_ = {
    'encoder_inputs': e_inp,
    'time_stamp_src': time_src,
    'src_mask': src_mask,
    'decoder_inputs': d_inp,
    'time_stamp_tgt': time_tgt,

    'original_data': org,
    'decoder_outputs': d_out,
    'translation': trans,
    'rotation': rot
  }

  return batch_

########################## CMU-Mocap Walking Dataset #######################

class CMUWalkingDataset(torch.utils.data.Dataset):
  
  def __init__(self, 
              params=None,
              mode='train', 
              **kwargs):
    super(CMUWalkingDataset, self).__init__(**kwargs)
    self._params = params
    self._data = {}
    self._mode = mode
    print('[INFO] (CMUWalkingDataset) mode: {}'.format(self._mode))
    self._framerates = {} # frq: stepsize
    self._selectedfrq = self._params['selected_frequency']
    self.load_data() 
    self._data_keys = self._data.keys()

  def collectfrq(self, org_frq):
    """Collect all possible framerates from original framerate."""
    
    min_frq = 4
    max_frq = 50
    for divisor in range(1, org_frq + 1):
        gcd = math.gcd(org_frq, divisor)
        frequency = int(org_frq // gcd)
        if frequency >= min_frq and frequency < max_frq and frequency not in self._framerates.keys():
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
    self._params['data_path'] = 'cmu_mocap'

    if self._mode == 'train' :
        dataset_path = glob.glob(self._params['data_path']  + '/' + 'train' + '/*')
    else :
        dataset_path = glob.glob(self._params['data_path']  + '/' + 'test' + '/*')

    # Load the YAML file
    with open(self._params['data_path'] + '/cmu_mocap.yaml', 'r') as yaml_file:
        dataset_attributes = yaml.safe_load(yaml_file)

    # Access the variables
    self._num_joints = dataset_attributes['_NMAJOR_JOINTS']
    self._pose_dim = 3*self._num_joints
    self._bone_joint1_idx = dataset_attributes['bone_joint1_idx']
    self._bone_joint2_idx = dataset_attributes['bone_joint2_idx']
    self._joint_names = dataset_attributes['joint_names']
    self._joint_edges = dataset_attributes['joint_edges']

    self._frq = dataset_attributes['frq']
    self._red_src = dataset_attributes['source_seq_len'] #25 = 0.5s
    self._red_tgt = dataset_attributes['target_seq_len'] #100 = 2s 
    
    self.collectfrq(self._frq)    
    self._source_seq_len = 5 # int(self._red_src/self._framerates[self._selectedfrq]) #5
    self._target_seq_len = 10 # int(self._red_tgt/self._framerates[self._selectedfrq]) #20

    all_dataset = []
    self.data_sequence_idx = []
    for file_ in dataset_path:
      action_sequence = np.load(file_)
      n_frames, dim = action_sequence.shape 

      entry_key = str(file_.split('/')[-1].split('.')[0])
      self._data[entry_key] = {}
      # self._data[entry_key] = np.zeros_like(action_sequence)
      # self._data[entry_key][:, :2] = action_sequence[:, :2]
      reformed_sequence = action_sequence[:, 2:].reshape(-1, 
                          self._num_joints, 3)[:, :, [0, 2, 1]].reshape(-1, 
                          self._num_joints*3)
      # reformed_sequence = self.preprocess_sequence(reformed_sequence)
      self._data[entry_key] = reformed_sequence  * (1.0/0.45)*2.54/100.0

      all_dataset.append(reformed_sequence)

      valid_frames = np.arange(0, n_frames - self._red_src - self._red_tgt + 1, 1)
      self.data_sequence_idx.extend(zip([entry_key] * len(valid_frames), 
                                        valid_frames.tolist()))

    all_dataset = np.concatenate(all_dataset, axis=0)
    print('[INFO] ({}) Dataset size: {}'.format(self.__class__.__name__, 
                                                all_dataset.shape))

    # # Access the variables
    # self._num_joints = len(_MAJOR_JOINTS)
    # self._pose_dim = 3*self._num_joints
    # self._bone_joint1_idx = [0, 0, 0, 1, 5, 6, 7, 8, 3, 9, 6, 11, 12, 13, 13, 13]
    # self._bone_joint2_idx = [1, 5, 6, 2, 4, 7, 8, 3, 9, 10, 11, 12, 13, 14, 15, 16]
    # self._joint_names = ['root', 'lhipjoint', 'lfemur', 'ltoes', 'rhipjoint', 
    #                     'rfemur', 'lowerback', 'upperback', 'thorax', 'lowerneck', 
    #                     'head', 'lclavicle', 'lhumerus', 'rclavicle', 'rhumerus', 'rradius']

    # suffle self.data_sequence_idx items
    np.random.shuffle(self.data_sequence_idx)
    np.random.shuffle(self.data_sequence_idx)

  ########################### other helper functions ############################
  def __len__(self):
    return len(self.data_sequence_idx) 

  def get_pose_dim(self):
    """Returns the pose dimension as a flattened vector."""
    return self._pose_dim
  
  ######################## get data item ###################################
  
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
    random_number = np.random.randint(-max_translation, max_translation)
    add_tensor = np.array([random_number, random_number, 0])
    
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
    translation = -output_frame[0] #pelvis 3
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
    delta_x = data_sel[source_seq_len, 0, 0] - data_sel[0, 0, 0]
    delta_y = data_sel[source_seq_len, 0, 1] - data_sel[0, 0, 1]
    angle = np.arctan2(delta_y, delta_x)

    # Create a rotation matrix around the z-axis and rotate
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle), 0],
                                [np.sin(-angle), np.cos(-angle), 0],
                                [0, 0, 1]])
    for t in range(T):
        rotated_sequence[t] = np.dot(rotation_matrix, data_sel[t].T).T   
    return rotated_sequence.reshape(T, self._num_joints*3), angle

  def __getitem__(self, index):
    """Get item for the training mode."""
    
    pose_size = self._pose_dim
    source_seq_len = self._source_seq_len
    target_seq_len = self._target_seq_len 
    total_frames = source_seq_len + target_seq_len

    idx, start_frame = self.data_sequence_idx[index]

    data_sel_full = self._data[idx][start_frame:(start_frame + self._red_src + self._red_tgt), 
                                    :]
    data_sel = data_sel_full[::3]
    T, _ = data_sel.shape 

    #original_data = data_sel.copy()
    interpolation = self._data[idx][start_frame:(start_frame+source_seq_len), 1]
    time_stamps_raw = self._data[idx][start_frame:(start_frame+total_frames), 0]
    time_stamps = time_stamps_raw - time_stamps_raw[0] # normalise time_stamps

    # Apply random translation and rotation during testing
    if self._params['random_transrot']:
      data_sel = self.apply_random_rotation(data_sel, 3.14)
      data_sel = self.apply_random_translation(data_sel, 10)
    original_data = data_sel.copy()

    # Transform to origin before training
    translation = np.zeros(3)
    rotation = np.zeros(1)
    if self._params['transform_torigin']:
      data_sel, translation = self.traslate_to_origin(data_sel, source_seq_len-1)
      data_sel, rotation = self.rotate_to_posxaxis(data_sel, source_seq_len-1)

    if self._params['missing_frames']:
      # randomly add 1's to indicate interpolated data
      max_ones = source_seq_len // 2  # Maximum number of ones
      num_ones = np.random.randint(0, max_ones + 1)  # Random number of ones to add
      indices = np.random.choice(source_seq_len, 
                                 num_ones, replace=False)  # Randomly select indices
      interpolation[indices] = 1  # Set selected indices to 1

    # select encoder input frames and time format
    indices_to_drop = np.where(interpolation == 1.)[0]
    enc_inputs = data_sel[:source_seq_len, :][interpolation == 0.].astype(np.float32)
    
    # INITIALIZE
    encoder_inputs = np.zeros((source_seq_len, pose_size), dtype=np.float32)
    time_stamps_src = np.zeros((source_seq_len), dtype=np.float32)
    interpolation = np.zeros((source_seq_len), dtype=np.float32)
    decoder_inputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)
    decoder_outputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)
    time_stamps_tgt = np.zeros((target_seq_len), dtype=np.float32)

    if self._params['missing_frames']:
      encoder_inputs[:enc_inputs.shape[0], :] = enc_inputs
      decoder_inputs = np.repeat(enc_inputs[-1:, :], 
                                 target_seq_len, 
                                 axis=0).astype(np.float32) # last input frame repeated
      time_stamps_src[:enc_inputs.shape[0]] = np.delete(time_stamps[:source_seq_len], 
                                                        indices_to_drop)
      interpolation[enc_inputs.shape[0]:] = 1
    else:
      encoder_inputs[:source_seq_len, :] = data_sel[:source_seq_len, :]
      decoder_inputs[:target_seq_len, :] = np.repeat(encoder_inputs[-1:, :], 
                                                     target_seq_len, axis=0)
      time_stamps_src[:source_seq_len] = time_stamps[:source_seq_len]

    time_stamps_tgt = time_stamps[source_seq_len:]
    decoder_outputs = data_sel[source_seq_len:, :] 

    return {
        'encoder_inputs': encoder_inputs,
        'time_stamp_src': time_stamps_src,
        'src_mask': interpolation,
        'decoder_inputs': decoder_inputs,
        'time_stamp_tgt': time_stamps_tgt,
        
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
    
    ax.set_xlim(-2 , 2 )
    ax.set_ylim(-2 , 2 )
    ax.set_zlim(0 , 2 )  
    ax.set_xlabel('X', labelpad=40)
    ax.set_ylabel('Y', labelpad=40)
    ax.set_zlabel('Z')

    ax.set_xticks(np.arange(-2, 2, 1)) 
    ax.set_yticks(np.arange(-2, 2, 1))  
    ax.set_zticks(np.arange( 0, 2, 5)) 

    lines = []
    scatter = None
    frames =[]
    for i in range(frames_input):
      
      joints = motion_input_[i,:,:] 
          
      # Plot the skeleton joints
      scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                           c='darkgreen', marker='o', s=3)
    
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
                           c='darkgreen', marker='o', s=3)
      scatter2 = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], 
                            c='darkred', marker='o', s=3)
      
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

  ################################ extra debugging functions ###############################
  def aug_test(self, index):
    
    source_seq_len = self._source_seq_len
    target_seq_len = self._target_seq_len 
    total_frames = source_seq_len + target_seq_len

    idx, start_frame = self.data_sequence_idx[index]

    data_sel_full = self._data[idx][start_frame:(start_frame + self._red_src + self._red_tgt), 2:]
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

    ax.set_xlim(-20 , 20 )
    ax.set_ylim(-20 , 20 )
    ax.set_zlim(-20 , 20 )  
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

    ax.set_xlim(-20 , 20 )
    ax.set_ylim(-20 , 20 )
    ax.set_zlim(-20 , 20 )  
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


############################## Dataset Factory ###########################

def dataset_factory(params):
  """Defines the datasets that will be used for training and validation."""

  train_dataset = CMUWalkingDataset(params, mode='train')
  train_dataset_fn = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=params['batch_size'],
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn
  )

  eval_dataset = CMUWalkingDataset(params, mode='eval')
  eval_dataset_fn = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=params['batch_size'],
    shuffle=False,
    drop_last=True,
    collate_fn=collate_fn
  ) 

  return train_dataset_fn, eval_dataset_fn

#################### Main Function #######################################

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', 
                      type=int, default=10)
  parser.add_argument('--transform_torigin', 
                      action='store_true', default=True)
  parser.add_argument('--random_transrot', 
                      action='store_true', default=False)
  parser.add_argument('--missing_frames', 
                      action='store_true', default=False)
  parser.add_argument('--selected_frequency', 
                      type=int, default=10)

  args = parser.parse_args()
  params = vars(args)

  dataset_t, dataset_e = dataset_factory(params)
  print('train loader size: {}'.format(len(dataset_t)))
  print('eval loader size: {}'.format(len(dataset_e)))

  data_class = CMUWalkingDataset(params, mode='train')
  for current_step, data in enumerate(dataset_t):
    i=9
    print(data['encoder_inputs'][i].shape)
    print(data['decoder_inputs'][i].shape)
    print(data['decoder_outputs'][i].shape)
    print(data['original_data'][i].shape)

    getdata= data['encoder_inputs'][i]
    getdataout = data['decoder_outputs'][i]
    getdatainp = data['decoder_inputs'][i]
    break
 
  # test, test_aug = data_class.aug_test(70)
  # print(test.shape, test_aug.shape)
  # data_class.plot_augumentation(test, test_aug)

  dataset_t.dataset.plot_predictions(getdata, 
                                     getdatainp, getdataout)

  data_class.plot_dataset(getdata)
  data_class.plot_dataset(getdatainp)
  data_class.plot_dataset(getdataout)
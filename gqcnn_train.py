import numpy as np
import random
import argparse
import json
import tensorflow as tf
import re
import os
import multiprocessing as mp
#import cv2
from gqcnn_network import GQCNN, reset_graph

class TensorDataset(object):
    def __init__(self, filename):
<<<<<<< HEAD
        self._filename = filename
        #self._filename = "/home/cjg429/Desktop/gqcnn-master/data/training/data/training/mini-dexnet_fc_pj_10_02_18/grasps"
=======
        #self._filename = filename
        self._filename = "/home/cjg429/Desktop/gqcnn-master/data/training/data/training/mini-dexnet_fc_pj_10_02_18/grasps"
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
        #self._config = json.load(open(config_filename, 'r'))

        self._im_field_name = "tf_depth_ims"
        self._pose_field_name = "grasps"
        self._label_field_name = "grasp_metrics"
        self._im_dataset = self.load(self._im_field_name)
        self._pose_dataset = self.load(self._pose_field_name)
        self._label_dataset = self.load(self._label_field_name)
        
        self._num_tensors = len(self._im_dataset)
        self._datapoints_per_file = self._im_dataset[0].shape[0]
        self._num_datapoints_last_file = self._im_dataset[self._num_tensors - 1].shape[0]
        self._num_datapoints = (self._num_tensors - 1) * self._datapoints_per_file + self._num_datapoints_last_file
        self._num_random_files = 10000
        self._num_random_files = min(self._num_tensors, self._num_random_files)
        self._max_files_eval = 1000
        
        self._index_to_file_num = {}
        self._file_num_to_indices = {}
        
        cur_file_num = 0
        start_datapoint_index = 0
        self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index
        for ind in range(self._num_datapoints):
            if ind > 0 and ind % self._datapoints_per_file == 0:
                cur_file_num += 1
                start_datapoint_index += self._datapoints_per_file
                if cur_file_num < self._num_tensors - 1:
                    self._file_num_to_indices[cur_file_num] = np.arange(self._datapoints_per_file) + start_datapoint_index
                else:
                    self._file_num_to_indices[cur_file_num] = np.arange(self._num_datapoints_last_file) + start_datapoint_index
            self._index_to_file_num[ind] = cur_file_num
        
<<<<<<< HEAD
        self._split_name = "image_wise"
=======
        self._split_name = "image_wise_1"
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
        self.split()
        self.compute_data_metrics()
            
        self._train_pct = 0.8
        self._total_pct = 1.0
        self._train_batch_size = 64
        self._angular_bins = 0
        if self._angular_bins != 0:
            self._max_angle = 123
            self._bin_width = 123
        self._pos_weight = 0.0
        if self._pos_weight != 0.0:
            self._neg_accept_prob = 123
            self._pos_accept_prob = 123
        self._metric_thresh = 0.5
        self._max_training_examples_per_load = 128
        self._im_height = 96
        self._im_width = 96
        self._im_channels = 1
        self._pose_dim = 1
        
    def datapoint_indices_for_tensor(self, tensor_index):
        if tensor_index > self._num_tensors:
            raise ValueError('Tensor index %d is greater than the number of tensors (%d)' %(tensor_index, self._num_tensors))
        return self._file_num_to_indices[tensor_index]
    
    def tensor_index(self, datapoint_index):
        if datapoint_index >= self._num_datapoints:
            raise ValueError('Datapoint index %d is greater than the number of datapoints (%d)' %(datapoint_index, self._num_datapoints))
        return self._index_to_file_num[datapoint_index]

    def split(self):
        indices_dir = os.path.join(self._filename, "splits", self._split_name)
        self._train_indices = np.load(os.path.join(indices_dir, "train_indices.npz"))["arr_0"]
        self._val_indices = np.load(os.path.join(indices_dir, "val_indices.npz"))["arr_0"]
        
        # loop through tensors, assigning indices to each file
        self._train_index_map = {}
        for i in range(self._num_tensors):
            self._train_index_map[i] = []
        
        for i in self._train_indices:
            tensor_index = self.tensor_index(i)
            datapoint_indices = self.datapoint_indices_for_tensor(tensor_index)
            lowest = np.min(datapoint_indices)
            self._train_index_map[tensor_index].append(i - lowest)
            
        for i, indices in self._train_index_map.items():
            self._train_index_map[i] = np.array(indices)
            
        self._val_index_map = {}
        for i in range(self._num_tensors):
            self._val_index_map[i] = []
            
        for i in self._val_indices:
            tensor_index = self.tensor_index(i)
            if tensor_index not in self._val_index_map.keys():
                self._val_index_map[tensor_index] = []
            datapoint_indices = self.datapoint_indices_for_tensor(tensor_index)
            lowest = np.min(datapoint_indices)
            self._val_index_map[tensor_index].append(i - lowest)

        for i, indices in self._val_index_map.items():
            self._val_index_map[i] = np.array(indices)
            
    def load(self, field_name):
        filename = os.path.join(self._filename, "tensors")
        raw_data_list = []
        filelist = [f for f in os.listdir(filename) if re.match(field_name + "_0", f)]
        filelist.sort()
        for i in range(0, len(filelist)):
            raw_data = np.load(os.path.join(filename, filelist[i]))["arr_0"]
            raw_data_list.append(raw_data)
        print(field_name + " file is loaded", len(raw_data_list))    
        return raw_data_list
    
    def tensor(self, field_name, file_num):
        if field_name == self._im_field_name:
            return self._im_dataset[file_num]
        elif field_name == self._pose_field_name:
            return self._pose_dataset[file_num]
        elif field_name == self._label_field_name:
            return self._label_dataset[file_num]
        else:
            raise ValueError('Field name %s not currently supported!' %(field_name))

    def datapoint(self, ind, field_names=None):
        datapoint = {}
        for field_name in field_names:
            datapoint[field_name] = None
            
        file_num = self._index_to_file_num[ind]
        for field_name in field_names:
            tensor = self.tensor(field_name, file_num)
            tensor_index = ind % self._datapoints_per_file
            datapoint[field_name] = tensor[tensor_index]
        return datapoint

    def compute_data_metrics(self):
        im_mean_filename = os.path.join(self._filename, 'im_mean.npy')
        im_std_filename = os.path.join(self._filename, 'im_std.npy')
        if os.path.exists(im_mean_filename) and os.path.exists(im_std_filename):
            self._im_mean = np.load(im_mean_filename)
            self._im_std = np.load(im_std_filename)
        else:
            self._im_mean = 0
            self._im_std = 0
            
            print("Comupting image mean and std")
            num_summed = 0

            for i in range(self._num_tensors):
                im_data = self.tensor(self._im_field_name, i)
                train_indices = self._train_index_map[i]
                if train_indices.shape[0] > 0:
                    self._im_mean += np.sum(im_data[train_indices, ...])
                    num_summed += train_indices.shape[0] * im_data.shape[1] * im_data.shape[2]
            self._im_mean = self._im_mean / num_summed

            for i in range(self._num_tensors):
                im_data = self.tensor(self._im_field_name, i)
                train_indices = self._train_index_map[i]
                if train_indices.shape[0] > 0:
                    self._im_std += np.sum((im_data[train_indices, ...] - self._im_mean) ** 2)
            self._im_std = np.sqrt(self._im_std / num_summed)

            np.save(os.path.join(self._filename, "im_mean.npy"), self._im_mean)
            np.save(os.path.join(self._filename, "im_std.npy"), self._im_std)
        
        pose_mean_filename = os.path.join(self._filename, 'pose_mean.npy')
        pose_std_filename = os.path.join(self._filename, 'pose_std.npy')
        if os.path.exists(pose_mean_filename) and os.path.exists(pose_std_filename):
            self._pose_mean = np.load(pose_mean_filename)
            self._pose_std = np.load(pose_std_filename)
        else:   
            self._pose_mean = 0
            self._pose_std = 0
            
            print("Comupting pose mean and std")
            num_summed = 0

            for i in range(self._num_tensors):
                pose_data = self.tensor(self._pose_field_name, i)
                train_indices = self._train_index_map[i]
                if train_indices.shape[0] > 0:
                    self._pose_mean += np.sum(read_pose_data(pose_data[train_indices, :]))
                    num_summed += train_indices.shape[0]
            self._pose_mean = self._pose_mean / num_summed

            for i in range(self._num_tensors):
                pose_data = self.tensor(self._pose_field_name, i)
                train_indices = self._train_index_map[i]
                if train_indices.shape[0] > 0:
                    self._pose_std += np.sum((read_pose_data(pose_data[train_indices, :]) - self._pose_mean) ** 2)
            self._pose_std = np.sqrt(self._pose_std / num_summed)

            np.save(os.path.join(self._filename, "pose_mean.npy"), self._pose_mean)
            np.save(os.path.join(self._filename, "pose_std.npy"), self._pose_std)
        
    def sample(self, norm_inputs=True):
        num_tensors = self._num_tensors
        train_index_map = self._train_index_map
        train_batch_size = self._train_batch_size

        angular_bins = self._angular_bins
        if angular_bins != 0:
            max_angle = self._max_angle
            bin_width = self._max_angle

        label_field_name = self._label_field_name

        pos_weight = self._pos_weight
        if pos_weight != 0.0:
            neg_accept_prob = self._neg_accept_prob
            pos_accept_prob = self._pos_accept_prob

        metric_thresh = self._metric_thresh
        max_training_examples_per_load = self._max_training_examples_per_load

        #norm_inputs = True
        im_mean = self._im_mean
        im_std = self._im_std
        pose_mean = self._pose_mean
        pose_std = self._pose_std

        im_field_name = self._im_field_name
        im_height = self._im_height
        im_width = self._im_width
        im_channels = self._im_channels

        pose_field_name = self._pose_field_name
        pose_dim = self._pose_dim

        train_images = np.zeros((train_batch_size, im_height, im_width, im_channels), dtype=np.float32)
        train_poses = np.zeros((train_batch_size, pose_dim), dtype=np.float32)
        train_labels = np.zeros((train_batch_size), dtype=np.int32)

        num_queued = 0
        start_i = 0
        end_i = 0
        file_num = 0
        
        if angular_bins > 0:
            train_pred_mask = np.zeros((train_batch_size, 2 * angular_bins), dtype=bool)

        while start_i < train_batch_size:
            num_remaining = train_batch_size - num_queued

            file_num = np.random.choice(num_tensors, size=1)[0]
            train_images_tensor = self.tensor(im_field_name, file_num)
            train_poses_tensor = self.tensor(pose_field_name, file_num)
            train_labels_tensor = self.tensor(label_field_name, file_num)

            train_ind = train_index_map[file_num]
            np.random.shuffle(train_ind)

            # filter positives and negatives
            if pos_weight != 0.0:
                labels = 1 * (train_labels_tensor > metric_thresh)
                np.random.shuffle(train_ind)
                filtered_ind = []
                for index in train_ind:
                    if labels[index] == 0 and np.random.rand() < neg_accept_prob:
                        filtered_ind.append(index)
                    elif labels[index] == 1 and np.random.rand() < pos_accept_prob:
                        filtered_ind.append(index)
                train_ind = np.array(filtered_ind)

            # samples train indices
            upper = min(num_remaining, train_ind.shape[0], max_training_examples_per_load)
            ind = train_ind[:upper]
            num_loaded = ind.shape[0]
            if num_loaded == 0:
                print("Queueing zero examples!!!!")
                continue

            # subsample data
            train_images_arr = train_images_tensor[ind, ...]
            train_poses_arr = train_poses_tensor[ind, ...]
            angles = train_poses_arr[:, 3]
            train_label_arr = train_labels_tensor[ind]
            num_images = train_images_arr.shape[0]

            # resize images
            rescale_factor = float(im_height) / train_images_arr.shape[1]
            if rescale_factor != 1.0:
                resized_train_images_arr = np.zeros((num_images, im_height, im_width, im_channels), dtype=np.float32)
                for i in range(num_images):
                    for c in range(train_images_arr.shape[3]):
                        resized_train_images_arr[i, :, :, c] = sm.imresize(train_images_arr[i, :, :, c], rescale_factor,
                                                                           interp='bicubic', mode='F')
                train_images_arr = resized_train_images_arr

            # add noises to images
            train_images_arr, train_poses_arr = distort(train_images_arr, train_poses_arr)

            # slice poses
            train_poses_arr = read_pose_data(train_poses_arr)

            # standardize inputs and outputs
            if norm_inputs:
                train_images_arr = (train_images_arr - im_mean) / im_std
                train_poses_arr = (train_poses_arr - pose_mean) / pose_std

            train_label_arr = 1 * (train_label_arr > metric_thresh)
            train_label_arr = train_label_arr.astype(bool)

            if angular_bins > 0:
                bins = np.zeros_like(train_label_arr)
                # form prediction mask to use when calculating loss
                neg_ind = np.where(angles < 0)
                angles = np.abs(angles) % max_angle
                angles[neg_ind] *= -1
                g_90 = np.where(angles > (max_angle / 2))
                l_neg_90 = np.where(angles < (-1 * (max_angle / 2)))
                angles[g_90] -= max_angle
                angles[l_neg_90] += max_angle
                angles *= -1 # hack to fix reverse angle convention
                angles += max_angle / 2
                train_pred_mask_arr = np.zeros((train_label_arr.shape[0], 2 * angular_bins))
                for i in range(angles.shape[0]):
                    bins[i] = angles[i] // bin_width
                    train_pred_mask_arr[i, int((angles[i] // bin_width) * 2)] = 1
                    train_pred_mask_arr[i, int((angles[i] // bin_width) * 2 + 1)] = 1        

            # compute the number of examples loaded
            num_loaded = train_images_arr.shape[0]
            end_i = start_i + num_loaded

            # enqueue training data batch
            train_images[start_i:end_i, ...] = train_images_arr.copy()
            train_poses[start_i:end_i, :] = train_poses_arr.copy()
            train_labels[start_i:end_i] = train_label_arr.copy()
            if angular_bins > 0:
                train_pred_mask[start_i:end_i] = train_pred_mask_arr.copy()

            del train_images_arr
            del train_poses_arr
            del train_label_arr

            # update start index
            start_i = end_i
            num_queued += num_loaded

        return train_images, train_poses, train_labels
    
def read_pose_data(pose_arr):
    if pose_arr.ndim == 1:
        return pose_arr[2:3]
    else:
        return pose_arr[:, 2:3]

def distort(image_arr, pose_arr):
    """ Adds noise to a batch of images """
    # read params
    symmetrize = True
    num_images = image_arr.shape[0]
    im_height = image_arr.shape[1]
    im_width = image_arr.shape[2]
    im_center = np.array([float(im_height - 1) / 2, float(im_width - 1) / 2])
    
    if symmetrize:
        for i in range(num_images):
            train_image = image_arr[i, :, :, 0]
            # rotate with 50% probability
            if np.random.rand() < 0.5:
                theta = 180.0
                rot_map = np.rot90(train_image, k=2)
                train_image = rot_map
                #rot_map = cv2.getRotationMatrix2D(tuple(im_center), theta, 1)
                #train_image = cv2.warpAffine(train_image, rot_map, (im_height, im_width), flags=cv2.INTER_NEAREST)
                
            # reflect left right with 50% probability
            if np.random.rand() < 0.5:
                train_image = np.fliplr(train_image)
            # reflect up down with 50% probability
            if np.random.rand() < 0.5:
                train_image = np.flipud(train_image)
            
            image_arr[i, :, :, 0] = train_image
    return image_arr, pose_arr

def error_rate_in_batches(gqcnn, dataset, num_files_eval=None, validation_set=True):
    all_predictions = []
    all_labels = []

    im_field_name = dataset._im_field_name
    pose_field_name = dataset._pose_field_name
    label_field_name = dataset._label_field_name
    metric_thresh = dataset._metric_thresh
    angular_bins = dataset._angular_bins
    
    # subsample files
    file_indices = np.arange(dataset._num_tensors)
    if num_files_eval is None:
        num_files_eval = dataset._max_files_eval
    np.random.shuffle(file_indices)
    if dataset._max_files_eval is not None and num_files_eval > 0:
        file_indices = file_indices[:num_files_eval]

    for i in file_indices:
        images = dataset.tensor(im_field_name, i)
        poses = dataset.tensor(pose_field_name, i)
        raw_poses = np.array(poses, copy=True)
        labels = dataset.tensor(label_field_name, i)

        # if no datapoints from this file are in validation then just continue
        if validation_set:
            indices = dataset._val_index_map[i]
        else:
            indices = dataset._train_index_map[i]                    
        if len(indices) == 0:
            continue
        
        images = images[indices, ...]
        poses = read_pose_data(poses[indices, :])
        raw_poses = raw_poses[indices, :]
        labels = labels[indices]

        labels = 1 * (labels > metric_thresh)
        labels = labels.astype(np.uint8)

        if angular_bins > 0:
            # form mask to extract predictions from ground-truth angular bins
            angles = raw_poses[:, 3]
            neg_ind = np.where(angles < 0)
            angles = np.abs(angles) % dataset._max_angle
            angles[neg_ind] *= -1
            g_90 = np.where(angles > (dataset._max_angle / 2))
            l_neg_90 = np.where(angles < (-1 * (dataset._max_angle / 2)))
            angles[g_90] -= dataset._max_angle
            angles[l_neg_90] += dataset._max_angle
            angles *= -1 # hack to fix reverse angle convention
            angles += (dataset._max_angle / 2)
            pred_mask = np.zeros((labels.shape[0], dataset._angular_bins*2), dtype=bool)
            for i in range(angles.shape[0]):
                pred_mask[i, int((angles[i] // dataset._bin_width)*2)] = True
                pred_mask[i, int((angles[i] // dataset._bin_width)*2 + 1)] = True
                
        # get predictions
        predictions = gqcnn.predict(images, poses)

        if angular_bins > 0:
            predictions = predictions[pred_mask].reshape((-1, 2))            

        # update
        all_predictions.extend(predictions[:, 1].tolist())
        all_labels.extend(labels.tolist())
                            
        # clean up
        del images
        del poses

    # get learning result
    true_count = 0
    for i in range(0, len(all_predictions)):
        if all_labels[i] == 1 and all_predictions[i] > 0.5:
            true_count += 1
        elif all_labels[i] == 0 and all_predictions[i] < 0.5:
            true_count += 1
<<<<<<< HEAD
=======
    
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    print(len([i for i, x in enumerate(all_predictions) if x > 0.5]))
    print(len([i for i, x in enumerate(all_labels) if x == 1]))
    print(len([i for i, x in enumerate(all_predictions) if x < 0.5]))
    print(len([i for i, x in enumerate(all_labels) if x == 0]))
<<<<<<< HEAD
    return true_count, len(all_predictions)
=======
    return true_count / len(all_predictions)
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    

def main():
    reset_graph()
<<<<<<< HEAD
    DATA_DIR = "/home/scarab6/Desktop/gqcnn/data/training/example_pj"
    dataset = TensorDataset(DATA_DIR)
    
=======
    DATA_DIR = "/home/cjg429/Desktop/gqcnn-master/data/training/example_pj/tensors"
    dataset = TensorDataset(DATA_DIR)   
    
    total_length = 17241
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    
<<<<<<< HEAD
    base_lr = learning_rate
    decay_step_multiplier = 0.33
=======
    base_lr = 1e-3
    decay_step_multiplier = 0.5
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    decay_rate = 0.95
    num_train = 0
    for train_indices in dataset._train_index_map.values():
        num_train += train_indices.shape[0]
    decay_step = decay_step_multiplier * num_train
<<<<<<< HEAD
    #min_learning_rate = 1e-6
    
=======
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    train_batch_size = 64
    
    train_step = 0
    decayed_learning_rate = base_lr
    
<<<<<<< HEAD
    save_frequency = 100
    eval_frequency = 100
    
    print("train", "step", "loss", "lr")
    for epoch in range(NUM_EPOCH):
        num_batches = int(num_train / batch_size)
=======
    print("train", "step", "loss", "lr")
    for epoch in range(NUM_EPOCH):
        num_batches = int(total_length / batch_size)
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
        for idx in range(num_batches):
            images, poses, labels = dataset.sample()
            
            decayed_learning_rate = base_lr * pow(decay_rate, (train_step * train_batch_size / decay_step))
<<<<<<< HEAD
            #decayed_learning_rate = max(decayed_learning_rate, min_learning_rate)
=======
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
              
            feed = {gqcnn.input_im_node: images, gqcnn.input_pose_node: poses, 
                    gqcnn.input_label_node: labels, gqcnn.input_drop_rate_node: 0.0, gqcnn.learning_rate: decayed_learning_rate}
            
            (train_loss, train_step, _) = gqcnn.sess.run([gqcnn.loss, gqcnn.global_step, gqcnn.train_op], feed)
            
<<<<<<< HEAD
            if ((train_step + 1) % eval_frequency == 0):
                curr_epoch = epoch + (float(idx) / num_batches)
                print("epoch", "%0.2f"%curr_epoch, "step", (train_step + 1), decayed_learning_rate, train_loss)
            if ((train_step + 1) % save_frequency == 0):
                gqcnn.save_model(model_save_path)
                print("save_model")
            if ((train_step + 1) % 500 == 0):
=======
            if ((train_step + 1) % 10 == 0):
                print("step", (train_step + 1), decayed_learning_rate, train_loss)
            if ((train_step + 1) % 5000 == 0):
                gqcnn.save_json(model_save_path + "/" + save_name + ".json")
                print("save_model")
                
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
                error_rate = error_rate_in_batches(gqcnn, dataset)
                print(error_rate)

    # finished, final model:
<<<<<<< HEAD
    gqcnn.save_model(model_save_path)
=======
    gqcnn.save_json(model_save_path + "/" + save_name + ".json")
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size', type=int, default=64)
    parser.add_argument(
<<<<<<< HEAD
        '--epochs', type=int, default=50)  
    parser.add_argument(
        '--lr', type=float, default=1e-2) 
=======
        '--epochs', type=int, default=2000000)  
    parser.add_argument(
        '--lr', type=float, default=1e-4) 
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    parser.add_argument(
        '--seed', type=int, default=1)         
    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    NUM_EPOCH = args.epochs
<<<<<<< HEAD
    save_name = "gqcnn_pretrain"
    model_save_path = "gqcnn_pretrain"
=======
    save_name = "gqcnn_full"
    model_save_path = "tf_gqcnn"
>>>>>>> d50452e82653da6fc8a059471974df4f57f72e37
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    gqcnn = GQCNN(is_training=True, reuse=True, gpu_mode=True)
    main()
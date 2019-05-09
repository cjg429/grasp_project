import numpy as np
import math
import json
import tensorflow as tf
import tensorflow.contrib.framework as tcf
import os
EPS = 1e-8

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    
class GQCNN(object):
    def __init__(self, batch_size=1, is_training=False, reuse=False, gpu_mode=False):
        self.batch_size = batch_size
        self.is_training = is_training
        self.momentum_rate = 0.9
        self.max_global_grad_norm = 100000000000
        self.reuse = reuse
        self.train_l2_regularizer = 0.0005
        
        model_dir = "/home/scarab6/Desktop/gqcnn/models/gqcnn_example_pj"
        #model_dir = "/home/cjg429/Desktop/gqcnn-master/data/training/data/training/mini-dexnet_fc_pj_10_02_18/grasps"
        im_mean_filename = os.path.join(model_dir, 'im_mean.npy')
        im_std_filename = os.path.join(model_dir, 'im_std.npy')
        self._im_mean = np.load(im_mean_filename)
        self._im_std = np.load(im_std_filename)

        pose_mean_filename = os.path.join(model_dir, 'pose_mean.npy')
        pose_std_filename = os.path.join(model_dir, 'pose_std.npy')
        self._pose_mean = np.load(pose_mean_filename)
        self._pose_std = np.load(pose_std_filename)
        self._batch_size = 64
        self._im_height = 96
        self._im_width = 96
        self._num_channels = 1
        self._pose_dim = 1
        
        self._weights = {}
        self._load_model = True
        self._pretrain = False
        
        with tf.variable_scope('gq_cnn', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.g = tf.Graph()
                    self._init_weights_file()
                    self._build_graph()
            else:
                tf.logging.info('Model using gpu.')
                self.g = tf.Graph()
                self._init_weights_file()
                self._build_graph()
        self._init_session()
    
    def _build_graph(self):
        #self.g = tf.Graph()
        with self.g.as_default():
            self.input_im_node = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])
            self.input_pose_node = tf.placeholder(tf.float32, shape=[None, 1])
            self.input_drop_rate_node = tf.placeholder_with_default(tf.constant(0.0), ())
            self.input_label_node = tf.placeholder(tf.int32, shape=[None])
            self.learning_rate = tf.placeholder_with_default(tf.constant(0.01), ())
            
            self._var_list = []
            def build_conv_layer(input_node, input_channels, filter_h, filter_w, num_filt, name, norm=False, pad='SAME'):
                with tf.name_scope(name):
                    if '{}_weights'.format(name) in self._weights.keys():
                        convW = self._weights['{}_weights'.format(name)]
                        convb = self._weights['{}_bias'.format(name)]
                    else:
                        convW_shape = [filter_h, filter_w, input_channels, num_filt]
                        fan_in = filter_h * filter_w * input_channels
                        std = np.sqrt(2.0 / (fan_in))
                        convW = tf.Variable(tf.truncated_normal(convW_shape, stddev=std), name='{}_weights'.format(name))
                        convb = tf.Variable(tf.truncated_normal([num_filt], stddev=std), name='{}_bias'.format(name))
                    convh = tf.nn.conv2d(input_node, convW, strides=[1, 1, 1, 1], padding=pad) + convb
                    convh = tf.nn.relu(convh)
                    return convh
            
            def build_fc_layer(input_node, fan_in, out_size, name, drop_rate, final_fc_layer=False):
                if '{}_weights'.format(name) in self._weights.keys():
                    fcW = self._weights['{}_weights'.format(name)]
                    fcb = self._weights['{}_bias'.format(name)]
                else:
                    std = np.sqrt(2.0 / (fan_in))
                    fcW = tf.Variable(tf.truncated_normal([fan_in, out_size], stddev=std), name='{}_weights'.format(name))
                    if final_fc_layer:
                        fcb = tf.Variable(tf.constant(0.0, shape=[out_size]), name='{}_bias'.format(name))
                    else:
                        fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))
                    self._var_list.append(fcW)
                    self._var_list.append(fcb)
                if final_fc_layer:
                    fc = tf.matmul(input_node, fcW) + fcb
                else:
                    fc = tf.nn.relu(tf.matmul(input_node, fcW) + fcb)
                
                fc = tf.nn.dropout(fc, 1 - drop_rate)
                return fc
            
            def build_pc_layer(input_node, fan_in, out_size, name):
                if '{}_weights'.format(name) in self._weights.keys():
                    pcW = self._weights['{}_weights'.format(name)]
                    pcb = self._weights['{}_bias'.format(name)]
                else:
                    std = np.sqrt(2.0 / (fan_in))
                    pcW = tf.Variable(tf.truncated_normal([fan_in, out_size], stddev=std), name='{}_weights'.format(name))
                    pcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))
                    self._var_list.append(pcW)
                    self._var_list.append(pcb)
                pc = tf.nn.relu(tf.matmul(input_node, pcW) + pcb)
                return pc
            
            def build_fc_merge(input_fc_node_1, input_fc_node_2, fan_in_1, fan_in_2, out_size, drop_rate, name):
                if '{}_input_1_weights'.format(name) in self._weights.keys():
                    input1W = self._weights['{}_input_1_weights'.format(name)]
                    input2W = self._weights['{}_input_2_weights'.format(name)]
                    fcb = self._weights['{}_bias'.format(name)] 
                else:
                    std = np.sqrt(2.0 / (fan_in_1 + fan_in_2))
                    input1W = tf.Variable(tf.truncated_normal([fan_in_1, out_size], stddev=std), name='{}_input_1_weights'.format(name))
                    input2W = tf.Variable(tf.truncated_normal([fan_in_2, out_size], stddev=std), name='{}_input_2_weights'.format(name))
                    fcb = tf.Variable(tf.truncated_normal([out_size], stddev=std), name='{}_bias'.format(name))
                    self._var_list.append(input1W)
                    self._var_list.append(input2W)
                    self._var_list.append(fcb)
            
                fc = tf.nn.relu(tf.matmul(input_fc_node_1, input1W) + tf.matmul(input_fc_node_2, input2W) + fcb)
                fc = tf.nn.dropout(fc, 1 - drop_rate)
                    
                return fc
            
            with tf.name_scope("im_stream"): # im_stream
                im_stream = build_conv_layer(self.input_im_node, 1, 9, 9, 16, "conv1_1", pad="VALID")
                im_stream = build_conv_layer(im_stream, 16, 5, 5, 16, "conv1_2", pad="VALID")
                im_stream = tf.nn.max_pool(im_stream, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                im_stream = build_conv_layer(im_stream, 16, 5, 5, 16, "conv2_1", pad="VALID")
                im_stream = build_conv_layer(im_stream, 16, 5, 5, 16, "conv2_2", pad="VALID")
                im_stream = tf.nn.max_pool(im_stream, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                im_stream = tf.reshape(im_stream, (-1, 17 * 17 * 16))
                im_stream = build_fc_layer(im_stream, 17 * 17 * 16, 64, "fc3", self.input_drop_rate_node, final_fc_layer=False)
            
            with tf.name_scope("pose_stream"):# pose_stream
                pose_stream = build_pc_layer(self.input_pose_node, 1, 16, "pc1")
            
            with tf.name_scope("merge_stream"):# merge_stream
                merge_stream = build_fc_merge(im_stream, pose_stream, 64, 16, 64, self.input_drop_rate_node, "fc4")
                merge_stream = build_fc_layer(merge_stream, 64, 2, "fc5", 0, final_fc_layer=True)
            
            self.output_tensor = tf.nn.softmax(merge_stream)
            
            # train ops
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.unregularized_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=self.input_label_node, logits=self.output_tensor, name=None))
                self.loss = self.unregularized_loss
                
                t_vars = tf.trainable_variables()
                self.regularizers = tf.nn.l2_loss(t_vars[0])
                for var in t_vars[1:]:
                    self.regularizers += tf.nn.l2_loss(var)
                self.loss += self.train_l2_regularizer * self.regularizers
                
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.loss, var_list=self._var_list))
                gradients, global_grad_norm = tf.clip_by_global_norm(gradients, self.max_global_grad_norm)
        # generate op to apply gradients
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

                #self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate)
                #grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.
                #self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
                '''
                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum_rate)
                grads = self.optimizer.compute_gradients(self.loss)
                grads, global_grad_norm = tf.clip_by_global_norm(grads, self.max_global_grad_norm)
                self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')'''

            # initialize vars
            self.init = tf.global_variables_initializer()

            t_vars = tf.trainable_variables()
            self.assign_ops = {}
            for var in t_vars:
            #if var.name.startswith('conv_vae'):
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)
                
    def _init_weights_file(self):
        if self._load_model == False:
            return
        
        if self._pretrain:
            self._base_layer_names = ["conv1_1_weights", "conv1_1_bias",
                                      "conv1_2_weights", "conv1_2_bias",
                                      "conv2_1_weights", "conv2_1_bias",
                                      "conv2_2_weights", "conv2_2_bias"]
        
        model_dir = "/home/scarab6/Desktop/gqcnn/models/gqcnn_example_pj"
        ckpt_file = os.path.join(model_dir, 'model.ckpt')
        
        with self.g.as_default():
            reader = tf.train.NewCheckpointReader(ckpt_file)
            ckpt_vars = tcf.list_variables(ckpt_file)
            full_var_names = []
            short_names = []
            self._weights = {}
            for variable, shape in ckpt_vars:
                full_var_names.append(variable)
                short_names.append(variable.split('/')[-1])
            # load variables
            if self._pretrain:
                for full_var_name, short_name in zip(full_var_names, short_names):
                    if short_name in self._base_layer_names:
                        self._weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)
                return
            
            for full_var_name, short_name in zip(full_var_names, short_names):
                self._weights[short_name] = tf.Variable(reader.get_tensor(full_var_name), name=full_var_name)
            
    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)
    
    def close_session(self):
        """ Close TensorFlow session """
        self.sess.close()
    
    '''def predict(self, image_arr, pose_arr):
        
        # setup for prediction
        num_batches = math.ceil(image_arr.shape[0] / self._batch_size)
        num_images = image_arr.shape[0]
        num_poses = pose_arr.shape[0]

        output_arr = None
        if num_images != num_poses:
            raise ValueError('Must provide same number of images as poses!')
            
        self._input_im_arr = (image_arr - self._im_mean) / self._im_std
        self._input_pose_arr = (pose_arr - self._pose_mean) / self._pose_std
        return self.sess.run(self.output_tensor, feed_dict={self.input_im_node: self._input_im_arr, self.input_pose_node: self._input_pose_arr})
    
        #self._input_im_arr = np.zeros((self._batch_size, self._im_height, self._im_width, self._num_channels))
        #self._input_pose_arr = np.zeros((self._batch_size, self._pose_dim))
                                        
        # predict in batches
        with self.g.as_default():
            if self.sess is None:
               raise RuntimeError('No TF Session open. Please call open_session() first.')
            i = 0
            batch_idx = 0
            while i < num_images:
                batch_idx += 1
                dim = min(self._batch_size, num_images - i)
                cur_ind = i
                end_ind = cur_ind + dim
                
                self._input_im_arr = (image_arr - self._im_mean) / self._im_std
                self._input_pose_arr = (pose_arr - self._pose_mean) / self._pose_std
                #self._input_im_arr[:dim, ...] = (image_arr[cur_ind:end_ind, ...] - self._im_mean) / self._im_std
                #self._input_pose_arr[:dim, :] = (pose_arr[cur_ind:end_ind, :] - self._pose_mean) / self._pose_std

                gqcnn_output = self.sess.run(self.output_tensor,
                                                feed_dict={self.input_im_node: self._input_im_arr,
                                                           self.input_pose_node: self._input_pose_arr})

                # allocate output tensor
                if output_arr is None:
                    output_arr = np.zeros([num_images] + list(gqcnn_output.shape[1:]))

                output_arr[cur_ind:end_ind, :] = gqcnn_output[:dim, :]
                i = end_ind

        return output_arr'''
    
    def predict(self, image_arr, pose_arr):
        input_im_arr = (image_arr - self._im_mean) / self._im_std
        input_pose_arr = (pose_arr - self._pose_mean) / self._pose_std
        return self.sess.run(self.output_tensor, feed_dict={self.input_im_node: input_im_arr, self.input_pose_node: input_pose_arr})
    
    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                #if var.name.startswith('conv_vae'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p*10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names
 
    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            #rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
        return rparam
                
    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                #if var.name.startswith('conv_vae'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
                idx += 1

    def load_json(self, jsonfile='gqcnn.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params)
  
    def save_json(self, jsonfile='gqcnn.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  
    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)
  
    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, os.path.join(model_save_path, 'model.ckpt'))
        '''checkpoint_path = os.path.join(model_save_path, 'model.ckpt')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0) # just keep one'''

    def load_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, os.path.join(model_save_path, 'model.ckpt'))
        
    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
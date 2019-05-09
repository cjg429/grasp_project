import numpy as np
import matplotlib.pyplot as plt
from gqcnn_train import TensorDataset, read_pose_data
from gqcnn_network import GQCNN, reset_graph

class Grasp2D(object):
    def __init__(self, center, angle=0.0, depth=1.0, width=0.0):
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width
        self.axis = np.array([np.cos(self.angle), np.sin(self.angle)]) 
        #self.center = Point(center, frame=frame)

class GQCNNAnalyzer(object):
    def __init__(self):
        model_dir = "/home/cjg429/Desktop/gqcnn-master/models/GQCNN-4.0-PJ"
    
    def analyze(self, output_dir=None):
        model_output_dir = output_dir
        train_result, val_result = self._run_prediction_single_model(model_output_dir)

    def _run_prediction_single_model(self, model_output_dir):
        gqcnn = GQCNN(is_training=False, reuse=True, gpu_mode=True)
        angular_bins = 0
        image_field_name = "tf_depth_ims"
        pose_field_name = "grasps"
        metric_name = "grasp_metrics"
        metric_thresh = 0.5
        
        dataset_dir = "/home/cjg429/Desktop/gqcnn-master/data/examples/single_object/primesense/depth_0.npy"
        dataset = TensorDataset(dataset_dir)
        train_indices = dataset._train_indices
        val_indices = dataset._val_indices
        
        all_predictions = []
        if angular_bins > 0:
            all_predictions_raw = []
        all_labels = []

        for i in range(dataset._num_tensors):
            image_arr = dataset.tensor(image_field_name, i)
            pose_arr = read_pose_data(dataset.tensor(pose_field_name, i))
            metric_arr = dataset.tensor(metric_name, i)
            label_arr = 1 * (metric_arr > metric_thresh)
            label_arr = label_arr.astype(np.uint8)
            if angular_bins > 0:
                # form mask to extract predictions from ground-truth angular bins
                raw_poses = dataset.tensor(pose_field_name, i)
                angles = raw_poses[:, 3]
                neg_ind = np.where(angles < 0)
                angles = np.abs(angles) % math.pi
                angles[neg_ind] *= -1
                g_90 = np.where(angles > (math.pi / 2))
                l_neg_90 = np.where(angles < (-1 * (math.pi / 2)))
                angles[g_90] -= math.pi
                angles[l_neg_90] += math.pi
                angles *= -1 # hack to fix reverse angle convention
                angles += (math.pi / 2)
                pred_mask = np.zeros((raw_poses.shape[0], angular_bins * 2), dtype=bool)
                bin_width = math.pi / angular_bins
                for i in range(angles.shape[0]):
                    pred_mask[i, int((angles[i] // bin_width)*2)] = True
                    pred_mask[i, int((angles[i] // bin_width)*2 + 1)] = True

            # predict with GQ-CNN
            predictions = gqcnn.predict(image_arr, pose_arr)
            if angular_bins > 0:
                raw_predictions = np.array(predictions)
                predictions = predictions[pred_mask].reshape((-1, 2))

            # aggregate
            all_predictions.extend(predictions[:,1].tolist())
            if angular_bins > 0:
                all_predictions_raw.extend(raw_predictions.tolist())
            all_labels.extend(label_arr.tolist())

        gqcnn.close_session()
        
        # create arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        train_predictions = all_predictions[train_indices]
        val_predictions = all_predictions[val_indices]
        train_labels = all_labels[train_indices]
        val_labels = all_labels[val_indices]
        if angular_bins > 0:
            all_predictions_raw = np.array(all_predictions_raw)
            train_predictions_raw = all_predictions_raw[train_indices]
            val_predictions_raw = all_predictions_raw[val_indices]        

        # aggregate results
        #train_result = BinaryClassificationResult(train_predictions, train_labels)
        #val_result = BinaryClassificationResult(val_predictions, val_labels)
        #train_result.save(os.path.join(model_output_dir, 'train_result.cres'))
        #val_result.save(os.path.join(model_output_dir, 'val_result.cres'))
        #true_positive_indices = val_result.true_positive_indices
        #np.random.shuffle(true_positive_indices)
        #true_positive_indices = true_positive_indices[:self.num_vis]
        for i, j in enumerate(true_positive_indices):
            k = val_indices[j]
            datapoint = dataset.datapoint(k, field_names=[image_field_name, pose_field_name])
            vis2d.clf()
            if angular_bins > 0:
                plot_grasp(datapoint, image_field_name, pose_field_name, angular_preds=val_predictions_raw[j])
            else: 
                plot_grasp(datapoint, image_field_name, pose_field_name)
            vis2d.title('Datapoint %d: Pred: %.3f Label: %.3f' %(k, val_result.pred_probs[j], 
                                                                 val_result.labels[j]), fontsize=self.font_size)
            vis2d.savefig(os.path.join(val_example_dir, 'true_positive_%03d.png' %(i)))

        return train_result, val_result
    
    def _plot_grasp(self, datapoint, image_field_name, pose_field_name, angular_preds=None):
        """ Plots a single grasp represented as a datapoint. """
        image = datapoint[image_field_name][:, :, 0]
        image_height = image.shape[0]
        image_width = image.shape[1]
        image_center = np.array([image_height / 2, image_width / 2])
        
        depth = datapoint[pose_field_name][2]
        width = 0
        grasps = []
        
        if angular_preds is not None:
            num_bins = angular_preds.shape[0] / 2
            bin_width = math.pi / num_bins
            for i in range(num_bins):
                bin_cent_ang = i * bin_width + bin_width / 2
                grasps.append(Grasp2D(center=image_center, angle=math.pi / 2 - bin_cent_ang, depth=depth, width=0.0))
            grasps.append(Grasp2D(center=image_center, angle=datapoint[pose_field_name][3], depth=depth, width=0.0))
        else:
            grasps.append(Grasp2D(center=image_center, angle=0, depth=depth, width=0.0))
        
        width = datapoint[pose_field_name][-1]
        
        vis2d.imshow(image)
        for i, grasp in enumerate(grasps[:-1]):
            vis2d.grasp(grasp, width=width, color=plt.cm.RdYlGn(angular_preds[i * 2 + 1]))
        vis2d.grasp(grasps[-1], width=width, color='b')
        
    def grasp(grasp, width=None, color='r', arrow_len=4, arrow_head_len = 2, arrow_head_width = 3,
              arrow_width = 1, jaw_len=3, jaw_width = 1.0,
              grasp_center_size=1, grasp_center_thickness=2.5,
              grasp_center_style='+', grasp_axis_width=1,
              grasp_axis_style='--', line_width=1.0, alpha=50, show_center=True, show_axis=False, scale=1.0):
        
        # set vars for suction
        skip_jaws = False
        if not hasattr(grasp, 'width'):
            grasp_center_style = '.'
            grasp_center_size = 50
            plt.scatter(grasp.center[0], grasp.center[1], c=color, marker=grasp_center_style, s=scale * grasp_center_size)

            if hasattr(grasp, 'orientation'):
                axis = np.array([np.cos(grasp.angle), np.sin(grasp.angle)])
                p = grasp.center + alpha * axis
                line = np.c_[grasp.center, p]
                plt.plot(line[0, :], line[1, :], color=color, linewidth=scale * grasp_axis_width)
                plt.scatter(p[0], p[1], c=color, marker=grasp_center_style, s=scale * grasp_center_size)
            return

        # plot grasp center
        if show_center:
            plt.plot(grasp.center[0], grasp.center[1], c=color, marker=grasp_center_style, mew=scale * grasp_center_thickness, ms=scale * grasp_center_size)
        if skip_jaws:
            return
        
        # compute axis and jaw locations
        axis = grasp.axis
        width_px = width
        if width_px is None and hasattr(grasp, 'width_px'):
            width_px = grasp.width_px
        g1 = grasp.center - (float(width_px) / 2) * axis
        g2 = grasp.center + (float(width_px) / 2) * axis
        g1p = g1 - scale * arrow_len * axis # start location of grasp jaw 1
        g2p = g2 + scale * arrow_len * axis # start location of grasp jaw 2

        # plot grasp axis
        if show_axis:
            plt.plot([g1[0], g2[0]], [g1[1], g2[1]], color=color, linewidth=scale * grasp_axis_width, linestyle=grasp_axis_style)
        
        # direction of jaw line
        jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])
        
        # length of arrow
        alpha = scale * (arrow_len - arrow_head_len)
        
        # plot first jaw
        g1_line = np.c_[g1p, g1 - scale * arrow_head_len * axis].T
        plt.arrow(g1p[0], g1p[1], alpha * axis[0], alpha * axis[1], width=scale * arrow_width, head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
        jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T

        plt.plot(jaw_line1[:, 0], jaw_line1[:, 1], linewidth=scale * jaw_width, c=color) 

        # plot second jaw
        g2_line = np.c_[g2p, g2 + scale * arrow_head_len * axis].T
        plt.arrow(g2p[0], g2p[1], -alpha * axis[0], -alpha * axis[1], width=scale * arrow_width, head_width=scale * arrow_head_width, head_length=scale * arrow_head_len, fc=color, ec=color)
        jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
        plt.plot(jaw_line2[:, 0], jaw_line2[:, 1], linewidth=scale * jaw_width, c=color) 
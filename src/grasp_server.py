#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.abspath('/home/user/graspnet_ws/graspnet-baseline')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

import rospy
import transformations as tf
from geometry_msgs.msg import Point, Pose, PoseStamped
from graspnet_ros.msg import Grasps
from graspnet_ros.srv import GetGrasps, GetGraspsResponse


def list_to_pose(pose_list):
    pose_msg = Pose()
    pose_msg.position.x = pose_list[0]
    pose_msg.position.y = pose_list[1]
    pose_msg.position.z = pose_list[2]
    pose_msg.orientation.x = pose_list[3]
    pose_msg.orientation.y = pose_list[4]
    pose_msg.orientation.z = pose_list[5]
    pose_msg.orientation.w = pose_list[6]
    return pose_msg


def matrix_to_pose(matrix):
    translation = list(tf.translation_from_matrix(matrix))
    quaternion = list(tf.quaternion_from_matrix(matrix))
    pose_list = translation + quaternion[1:] + quaternion[:1]
    # pose_list = translation + quaternion
    return list_to_pose(pose_list)


class GraspPlanner():

    def __init__(self):
        # TODO read gripper params
        self.hand_depth = 0.0613
        self.hand_height = 0.035

        # Init the model
        checkpoint_path = rospy.get_param(
            '~checkpoint_path',
            '/home/user/graspnet_ws/src/graspnet_ros/checkpoint-rs.tar'
            # '/home/user/graspnet_ws/src/graspnet_ros/minkuresunet_realsense.tar'
        )
        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(
            "-> loaded checkpoint {} (epoch: {})".format(
                checkpoint_path, start_epoch
            )
        )
        # set model to eval mode
        net.eval()
        self.net = net

    def get_grasp_poses(
        self,
        points,  # N x (x,y,z)
        colors,  # N x (r,g,b)
        num_samples=20000,
        vizualize=True
    ):
        # transform to trained coordinate frame
        frame_rotate = np.array(
            [
                [0, -1., 0., 0.],
                [-1, 0., 0., 0.],
                [0., 0, -1., 0.],
                [0., 0., 0., 1.],
            ]
        )
        points = (frame_rotate[:3, :3] @ points.T).T

        # sample points
        if len(points) >= num_samples:
            idxs = np.random.choice(len(points), num_samples, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(
                len(points), num_samples - len(points), replace=True
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = points[idxs]
        color_sampled = colors[idxs]

        # convert data
        end_points = dict()
        cloud_sampled = torch.from_numpy(
            cloud_sampled[np.newaxis].astype(np.float32)
        )
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        if vizualize:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
            cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
            gg.nms()
            gg.sort_by_score()
            gg = gg[:100]
            # gg0 = gg[:len(gg) // 2]
            # grippers = gg0.to_open3d_geometry_list()
            # o3d.visualization.draw_geometries([cloud, *grippers])
            # gg1 = gg[len(gg) // 2:]
            # grippers = gg1.to_open3d_geometry_list()
            # o3d.visualization.draw_geometries([cloud, *grippers])
            grippers = gg.to_open3d_geometry_list()
            geometries = [cloud, *grippers]
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            for geometry in geometries:
                viewer.add_geometry(geometry)
            opt = viewer.get_render_option()
            opt.show_coordinate_frame = True
            opt.background_color = np.asarray([0.5, 0.5, 0.5])
            viewer.run()
            viewer.destroy_window()

        pose_list = []
        score_list = []
        for grasp in gg:
            pose = np.eye(4)
            pose[:3, :3] = grasp.rotation_matrix
            pose[:3, 3] = grasp.translation

            # transform back to input frame
            pose = frame_rotate.T @ pose

            #flip x and z axis
            pose = np.matmul(
                pose,
                [
                    [0., 0., 1., 0.],
                    [0, -1., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.],
                ],
            )

            pose_list.append(matrix_to_pose(pose))
            score_list.append(grasp.score)
        return pose_list, score_list

    def handle_grasp_request(self, req):
        points = np.array([(p.x, p.y, p.z) for p in req.points])
        colors = np.array([(c.r, c.g, c.b) for c in req.colors])
        grasps, scores = self.get_grasp_poses(points, colors)
        grasps_msg = Grasps()
        grasps_msg.poses = grasps
        grasps_msg.scores = scores
        grasps_msg.samples = [Point(0, 0, 0) for p in grasps]  # TODO: implement

        return GetGraspsResponse(grasps_msg)


if __name__ == "__main__":
    rospy.init_node('graspnet_grasp_server')
    print(rospy.get_param_names())
    grasp_planner = GraspPlanner()
    s = rospy.Service(
        'get_grasps', GetGrasps, grasp_planner.handle_grasp_request
    )
    print("Ready to generate grasps...")
    rospy.spin()

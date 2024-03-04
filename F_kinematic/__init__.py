import torch
import torch.nn as nn
import pos_encoding
import torch.nn.functional as F
from utils import general_utils
from weightMLP import VanillaCondMLP

class RigidTransform(nn.Module):
    def __init__(self, num_joints, hidden_dim=3, num_layers=3, encoding=40, rotation_representation="quat"):
        super(RigidTransform, self).__init__()
        self.num_joints = num_joints
        self.num_part = num_joints + 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rotation_representation = rotation_representation
        if encoding is not None:
            self.encoding = pos_encoding.FixedPositionalEncoding(encoding)
            self.input_dim = self.num_joints + self.num_joints * encoding
        else:
            self.encoding = torch.nn.Identity()
            self.input_dim = self.num_joints

        if self.rotation_representation == "quat":
            # 以四元数表示旋转
            self.output_dim_rotation = self.num_part * 4
        elif self.rotation_representation == "mat":
            # 以6D矩阵表示旋转
            self.output_dim_rotation = self.num_part * 6

        rotation_net = []
        for i in range(num_layers):
            if i == 0:
                rotation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                rotation_net.append(nn.Linear(self.hidden_dim, self.output_dim_rotation))
            else:
                rotation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.rotation_net = nn.ModuleList(rotation_net)

        self.output_dim_translation = self.num_part * 3
        translation_net = []
        for i in range(num_layers):
            if i == 0:
                translation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                translation_net.append(nn.Linear(self.hidden_dim, self.output_dim_translation))
            else:
                translation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.translation_net = nn.ModuleList(translation_net)

    def forward(self, joints):
        # joints: [num_joints]
        joints = self.encoding(joints)
        rotation = joints
        translation = joints
        for i in range(self.num_layers):
            rotation = self.rotation_net[i](rotation)
            translation = self.translation_net[i](translation)
            if i < self.num_layers - 1:
                rotation = F.relu(rotation)
                translation = F.relu(translation)

        if self.rotation_representation == "quat":
            rotation = rotation.view(self.num_part, 4)
            rotation = general_utils.build_rotation(rotation)
        elif self.rotation_representation == "mat":
            rotation = rotation.view(self.num_part, 6)
            rotation = general_utils.rotation_6d_to_matrix(rotation)

        translation = translation.view(self.num_part, 3)
        # rotation: [num_part, 3, 3], translation: [num_part, 3]
        # convert to homogeneous transformation matrix [num_part, 4, 4]
        transformation = torch.zeros((self.num_part, 4, 4), device='cuda')
        transformation[:, :3, :3] = rotation
        transformation[:, :3, 3] = translation
        transformation[:, 3, 3] = 1.0
        return transformation


class GaussianTransformer(nn.Module):
    def __init__(self, num_joints, hidden_dim=3, num_layers=3, encoding=40, rotation_representation="quat"):
        super(GaussianTransformer, self).__init__()
        self.transformer = RigidTransform(num_joints, hidden_dim, num_layers, encoding, rotation_representation)
        self.num_joints = num_joints
        self.weight_net = VanillaCondMLP(3, 0, num_joints + 1)

    def forward(self, points, joints):
        # points: [num_points, 3]
        transformation = self.transformer(joints)
        # transformation: [num_joints + 1, 4, 4]
        weights = self.weight_net(points)
        # weights: [num_points, num_joints + 1]
        weights = F.softmax(weights, dim=-1)
        fwd = torch.matmul(weights, transformation.view(-1, 16)).view(-1, 4, 4)
        # fwd: [num_points, 4, 4]
        return fwd

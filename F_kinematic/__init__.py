import torch
import torch.nn as nn
from F_kinematic import pos_encoding
import torch.nn.functional as F
from utils import general_utils
from F_kinematic.weightMLP import VanillaCondMLP
from F_kinematic.embedding import SimpleEmbedding, EmbeddingDecomposition
from PointTransformer.model import PointTransformerSeg

class RigidTransform(nn.Module):
    def __init__(self, num_joints, hidden_dim=128, num_layers=2, encoding=20, rotation_representation="quat"):
        super(RigidTransform, self).__init__()
        self.num_joints = num_joints
        self.num_part = num_joints + 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rotation_representation = rotation_representation
        self.embedding_dim = 128
        if encoding is not None:
            self.encoding = pos_encoding.FixedPositionalEncoding(encoding)
            self.input_dim = self.num_joints + self.num_joints * encoding
        else:
            self.encoding = torch.nn.Identity()
            self.input_dim = self.num_joints

        self.joint_embedding = SimpleEmbedding(self.input_dim)
        self.structure_embedding = nn.Embedding(1, self.embedding_dim)
        self.decomposition = EmbeddingDecomposition(2 * self.embedding_dim, self.embedding_dim, self.num_part)

        if self.rotation_representation == "quat":
            # 以四元数表示旋转
            self.output_dim_rotation = 4 * self.num_part
        elif self.rotation_representation == "mat":
            # 以6D矩阵表示旋转
            self.output_dim_rotation = 6 * self.num_part

        rotation_net = []
        for i in range(num_layers):
            if i == 0:
                rotation_net.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            elif i == num_layers - 1:
                rotation_net.append(nn.Linear(self.hidden_dim, self.output_dim_rotation))
            else:
                rotation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.rotation_net = nn.ModuleList(rotation_net)

        self.output_dim_translation = 3 * self.num_part
        translation_net = []
        for i in range(num_layers):
            if i == 0:
                translation_net.append(nn.Linear(self.embedding_dim, self.hidden_dim))
            elif i == num_layers - 1:
                translation_net.append(nn.Linear(self.hidden_dim, self.output_dim_translation))
            else:
                translation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.translation_net = nn.ModuleList(translation_net)

    def forward(self, joints):
        # joints: [num_joints]
        joints = self.encoding(joints)
        joints_embed = self.joint_embedding(joints)
        # structure_embed = self.structure_embedding(torch.zeros(1, dtype=torch.long, device=joints.device)).squeeze(0)
        # embed = torch.cat([joints_embed, structure_embed], dim=-1)
        # decomposed = self.decomposition(embed)

        rotation = joints_embed
        translation = joints_embed
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


class GaussianDeformer(nn.Module):
    def __init__(self, num_joints, hidden_dim=128, num_layers=2, encoding=20, rotation_representation="quat"):
        super(GaussianDeformer, self).__init__()
        self.rigid = RigidTransform(num_joints, hidden_dim, num_layers, encoding, rotation_representation)
        self.num_joints = num_joints
        # manually create cfg for PointTransformerSeg
        class Config:
            def __init__(self, num_point, num_class, input_dim):
                self.num_point = num_point
                self.model = self
                self.nblocks = 4
                self.nneighbor = 16
                self.num_class = num_class
                self.input_dim = input_dim
                self.transformer_dim = 128

        self.num_point = 1024
        num_class_for_pt = num_joints + 1
        input_dim_for_pt = 3 + 1
        cfg = Config(self.num_point, num_class_for_pt, input_dim_for_pt)
        self.point_transformer = PointTransformerSeg(cfg)
        self.optimizer_rigid, self.scheduler = None, None
        self.optimizer_pt = None
        # self.set_optimizer()

    def set_optimizer(self, max_iter):
        self.optimizer_rigid = torch.optim.Adam(self.rigid.parameters(), lr=1e-3, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_rigid, step_size=1000, gamma=0.5)
        self.optimizer_pt = torch.optim.Adam(self.point_transformer.parameters(), lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-4)


    def forward(self, points, joints):
        # points: [batch_size, num_point, 3]
        batch_size = points.size(0)
        transformation = self.rigid(joints)
        # transformation: [num_joints + 1, 4, 4]
        # create input for point transformer
        pt_input = torch.cat([points, torch.ones((points.size(0), points.size(1), 1), device=points.device)], dim=-1)
        weights = self.point_transformer(pt_input)

        # weights: [batch_size, num_point, num_joints + 1]
        weights = F.softmax(weights, dim=-1)
        weights = weights.view(batch_size * self.num_point, -1)
        # fwd = torch.matmul(weights, transformation.view(-1, 16)).view(-1, 4, 4)
        # # fwd: [batch_size * num_point, 4, 4]
        # fwd = fwd.reshape(batch_size, -1, 4, 4)
        belongs = torch.argmax(weights, dim=-1)
        fwd = transformation[None, :, :, :].repeat(batch_size * self.num_point, 1, 1, 1)
        # choose the corresponding transformation for each point
        fwd = fwd[torch.arange(batch_size * self.num_point), belongs]
        fwd = fwd.view(batch_size, -1, 4, 4)
        return fwd

    def optimize(self):
        self.optimizer_rigid.step()
        self.scheduler.step()
        self.optimizer_pt.step()
        self.optimizer_rigid.zero_grad()
        self.optimizer_pt.zero_grad()
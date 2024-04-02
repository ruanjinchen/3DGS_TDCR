import torch
import torch.nn as nn
from F_kinematic import pos_encoding
import torch.nn.functional as F
from utils import general_utils
from F_kinematic.weightMLP import VanillaCondMLP
from F_kinematic.embedding import SimpleEmbedding, EmbeddingDecomposition
from PointTransformer.model import PointTransformerSeg
from sklearn.cluster import KMeans
from pytorch3d.ops.knn import knn_points

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
        self.joints_embed = SimpleEmbedding(self.num_joints, self.embedding_dim)
        self.ellipsoid_point_encodings = pos_encoding.FixedPositionalEncoding(encoding)
        self.ellipsoid_input_dim = 3 + 3 * encoding
        self.input_dim = self.embedding_dim + self.ellipsoid_input_dim

        if self.rotation_representation == "quat":
            # 以四元数表示旋转
            self.output_dim_rotation = 4
        elif self.rotation_representation == "mat":
            # 以6D矩阵表示旋转
            self.output_dim_rotation = 6

        rotation_net = []
        for i in range(num_layers):
            if i == 0:
                rotation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                rotation_net.append(nn.Linear(self.hidden_dim, self.output_dim_rotation))
            else:
                rotation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.rotation_net = nn.ModuleList(rotation_net)

        self.output_dim_translation = 3
        translation_net = []
        for i in range(num_layers):
            if i == 0:
                translation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                translation_net.append(nn.Linear(self.hidden_dim, self.output_dim_translation))
            else:
                translation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.translation_net = nn.ModuleList(translation_net)

    def forward(self, joints, points):
        # joints: [num_joints]
        # joints = self.encoding(joints)
        joints = self.joints_embed(joints)
        # structure_embed = self.structure_embedding(torch.zeros(1, dtype=torch.long, device=joints.device)).squeeze(0)
        # embed = torch.cat([joints_embed, structure_embed], dim=-1)
        # decomposed = self.decomposition(embed)
        points = self.ellipsoid_point_encodings(points)
        joints_embed = torch.cat([joints, points], dim=-1)

        rotation = joints_embed
        translation = joints_embed
        for i in range(self.num_layers):
            rotation = self.rotation_net[i](rotation)
            translation = self.translation_net[i](translation)
            if i < self.num_layers - 1:
                rotation = F.relu(rotation)
                translation = F.relu(translation)

        if self.rotation_representation == "quat":
            rotation = rotation.view(4)
            rotation = general_utils.build_rotation(rotation)
        elif self.rotation_representation == "mat":
            rotation = rotation.view(6)
            rotation = general_utils.rotation_6d_to_matrix(rotation)

        translation = translation.view(3)
        # rotation: [num_part, 3, 3], translation: [num_part, 3]
        # convert to homogeneous transformation matrix [num_part, 4, 4]
        transformation = torch.zeros((4, 4), device='cuda')
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        transformation[3, 3] = 1.0
        return transformation

class RigidTransform_mul(nn.Module):
    def __init__(self, num_joints, hidden_dim=128, num_layers=2, encoding=20, rotation_representation="quat"):
        super(RigidTransform_mul, self).__init__()
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
        self.joints_embed = SimpleEmbedding(self.num_joints, self.embedding_dim)
        self.ellipsoid_point_encodings = pos_encoding.FixedPositionalEncoding(encoding)
        self.ellipsoid_input_dim = 3 + 3 * encoding
        self.input_dim = self.embedding_dim + self.ellipsoid_input_dim

        if self.rotation_representation == "quat":
            # 以四元数表示旋转
            self.output_dim_rotation = 4 * self.num_part
        elif self.rotation_representation == "mat":
            # 以6D矩阵表示旋转
            self.output_dim_rotation = 6 * self.num_part

        rotation_net = []
        for i in range(num_layers):
            if i == 0:
                rotation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                rotation_net.append(nn.Linear(self.hidden_dim, self.output_dim_rotation))
            else:
                rotation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.rotation_net = nn.ModuleList(rotation_net)

        self.output_dim_translation = 3 * self.num_part
        translation_net = []
        for i in range(num_layers):
            if i == 0:
                translation_net.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == num_layers - 1:
                translation_net.append(nn.Linear(self.hidden_dim, self.output_dim_translation))
            else:
                translation_net.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.translation_net = nn.ModuleList(translation_net)

    def forward(self, joints, points):
        # joints: [num_joints]
        # joints = self.encoding(joints)
        joints = self.joints_embed(joints)
        # structure_embed = self.structure_embedding(torch.zeros(1, dtype=torch.long, device=joints.device)).squeeze(0)
        # embed = torch.cat([joints_embed, structure_embed], dim=-1)
        # decomposed = self.decomposition(embed)
        points = self.ellipsoid_point_encodings(points)
        joints_embed = torch.cat([joints, points], dim=-1)

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
    def __init__(self, num_joints, num_part=None, hidden_dim=128, num_layers=5, encoding=20, rotation_representation="mat", use_mlp=False):
        super(GaussianDeformer, self).__init__()
        self.rigid = RigidTransform(num_joints, 256, 8, encoding, rotation_representation)
        self.num_joints = num_joints
        self.num_part = num_part if num_part is not None else num_joints + 1
        self.optimizer_rigid, self.scheduler = None, None
        # self.scale_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 3 * (num_joints + 1)),
        # )
        # self.bone_parameters = nn.Parameter(torch.randn((num_joints + 1, hidden_dim), dtype=torch.float, device='cuda'))
        self.point_embedding = pos_encoding.FixedPositionalEncoding(encoding)
        self.point_input_dim = 3 + 3 * encoding

        self.ellipsoid_center_point = torch.randn((self.num_part, 3), dtype=torch.float, device='cuda')
        self.ellipsoid_radii = torch.randn((self.num_part, 3), dtype=torch.float, device='cuda')
        self.ellipsoid_rotation = torch.randn((self.num_part, 6), dtype=torch.float, device='cuda') # 6D rotation

        # register ellipsoid inited as buffer
        self.register_buffer("ellipsoid_inited", torch.tensor(False))
        self.use_mlp = use_mlp
        if use_mlp:
            self.weight_mlp = VanillaCondMLP(self.point_input_dim, 0, self.num_part)

    def set_optimizer(self, max_steps=100000):
        self.optimizer_rigid = torch.optim.Adam(self.parameters(), lr=5e-4, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_rigid, max_steps, eta_min=1e-8)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_rigid, step_size=10000, gamma=0.5)
    def load(self, pa, optimizer=None, add=True):
        if add:
            self.ellipsoid_center_point = nn.Parameter(self.ellipsoid_center_point, requires_grad=True)
            self.ellipsoid_radii = nn.Parameter(self.ellipsoid_radii, requires_grad=True)
            self.ellipsoid_rotation = nn.Parameter(self.ellipsoid_rotation, requires_grad=True)
            if optimizer is not None:
                self.optimizer_rigid.add_param_group({"params": self.ellipsoid_center_point})
                self.optimizer_rigid.add_param_group({"params": self.ellipsoid_radii})
                self.optimizer_rigid.add_param_group({"params": self.ellipsoid_rotation})

        self.load_state_dict(pa)
        if optimizer is not None:
            self.optimizer_rigid.load_state_dict(optimizer)


    def set_ellipsoid(self, points, random=False):
        # if self.use_mlp:
        #     self.ellipsoid_inited = torch.tensor(True)
        #     return
        # use k-means to find the center of the ellipsoid
        if random:
            # random choose center from points
            center = points[torch.randint(0, points.size(0), (self.num_part,))]
            self.ellipsoid_center_point = nn.Parameter(center, requires_grad=True)
            self.ellipsoid_radii = nn.Parameter(torch.zeros((self.num_part, 3), dtype=torch.float, device='cuda'), requires_grad=True)
        else:
            kmeans = KMeans(n_clusters=self.num_part, random_state=0).fit(points.cpu().detach().numpy())
            ellipsoid_center_point = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device='cuda')
            self.ellipsoid_center_point = nn.Parameter(ellipsoid_center_point, requires_grad=True)
            # compute the radii of the ellipsoid
            distance = torch.cdist(points.detach(), ellipsoid_center_point, 2)
            distance = distance.cpu().numpy()
            # radii is the maximum distance from the center to the points belong to the ellipsoid
            radii = []
            for i in range(self.num_part):
                radii.append(distance[kmeans.labels_ == i][..., i].max())
            ellipsoid_radii = torch.tensor(radii, dtype=torch.float, device='cuda')

            # expand to [num_part, 3]
            ellipsoid_radii = ellipsoid_radii[:, None].repeat(1, 3)

            self.ellipsoid_radii = nn.Parameter(torch.log(ellipsoid_radii), requires_grad=True)

        self.ellipsoid_rotation = nn.Parameter(torch.randn((self.num_part, 6), dtype=torch.float, device='cuda'),
                                               requires_grad=True)
        # add to optimizer
        self.optimizer_rigid.add_param_group({"params": self.ellipsoid_center_point})
        self.optimizer_rigid.add_param_group({"params": self.ellipsoid_radii})
        self.optimizer_rigid.add_param_group({"params": self.ellipsoid_rotation})
        self.ellipsoid_inited = torch.tensor(True)

    def ellipsoid(self, points, rotation, translation):
        # points: [num_point, 3]
        # compute Mahalanobis distance from each point to the surface of each ellipsoid
        # [num_point, num_part]

        # build rotation matrix
        rotation = general_utils.rotation_6d_to_matrix(self.ellipsoid_rotation)
        # rotation: [num_part, 3, 3]
        # build radii matrix [num_part, 3, 3]
        radii = torch.exp(self.ellipsoid_radii)
        radii = torch.diag_embed(radii)
        # clamp radii to avoid numerical instability

        # w = (points - center)^T * R^T * D * R * (points - center)
        cord = points.unsqueeze(1) - self.ellipsoid_center_point.unsqueeze(0) # [num_point, num_part, 3]
        cord = cord.unsqueeze(-1) # [num_point, num_part, 3, 1]
        W = cord.transpose(-1, -2) @ rotation.transpose(-1, -2) @ radii @ rotation @ cord
        return W.squeeze()


    # def forward(self, points, joints):
    #     # points: [num_point, 3]
    #     # joints: [num_joints]
    #     n_points = points.size(0)
    #     # transformation = self.rigid(joints)
    #     transformation = []
    #     for i in range(self.num_part):
    #         transformation.append(self.rigid(joints, self.ellipsoid_center_point[i]))
    #     transformation = torch.stack(transformation, dim=0)
    #     # transformation: [num_part, 4, 4]
    #     homocord_center = torch.cat([self.ellipsoid_center_point, torch.ones((self.num_part, 1), dtype=torch.float, device='cuda')], dim=-1)
    #     transformed_center = torch.matmul(transformation, homocord_center.unsqueeze(-1)).squeeze(-1)
    #     transformed_center = transformed_center[:, :3]
    #
    #     if self.use_mlp:
    #         weights = self.weight_mlp(self.point_embedding(points))
    #         weights = torch.softmax(weights, dim=-1)
    #     else:
    #         weights = self.ellipsoid(points)
    #         weights = weights + 1e-6
    #         weights = weights / weights.sum(dim=-1, keepdim=True)
    #     # optinal: use mlp to directly estimate weights
    #
    #     # weights: [num_point, num_part]
    #
    #     fwd = torch.matmul(weights, transformation.view(-1, 16)).view(-1, 4, 4)
    #     # fwd: [num_point, 4, 4]
    #
    #     return fwd, transformed_center

    def forward(self, points, joints):
        transformation = []
        for i in range(self.num_part):
            transformation.append(self.rigid(joints, self.ellipsoid_center_point[i]))
        transformation = torch.stack(transformation, dim=0)
        # transformation = self.rigid(joints)
        rotation_obs = transformation[:, :3, :3]
        translation_obs = transformation[:, :3, 3]
        # transformation: [num_part, 4, 4]


        if self.use_mlp:
            weights = self.weight_mlp(self.point_embedding(points))
            weights = torch.softmax(weights, dim=-1)
            translation = translation_obs
            rotation = rotation_obs
            local_points = points.unsqueeze(1) - self.ellipsoid_center_point.unsqueeze(0)
            local_points_rotated = torch.matmul(rotation.unsqueeze(0), local_points.unsqueeze(-1)).squeeze(-1)
            points_transformed = local_points_rotated + self.ellipsoid_center_point.unsqueeze(
                0) + translation.unsqueeze(0)
            points_transformed = torch.sum(points_transformed * weights.unsqueeze(-1), dim=1)

            rotation = rotation[None, :, :, :].repeat(points.size(0), 1, 1, 1)
            rotation = torch.sum(rotation * weights.unsqueeze(-1).unsqueeze(-1), dim=1)
            # patch rotation to 4x4 for compatibility
            rotation = torch.cat([rotation, torch.zeros((points.size(0), 1, 3), dtype=torch.float, device='cuda')],
                                 dim=1)
            rotation = torch.cat([rotation, torch.zeros((points.size(0), 4, 1), dtype=torch.float, device='cuda')],
                                 dim=2)
            rotation[:, 3, 3] = 1.0

            center_transformed = self.ellipsoid_center_point + translation

            return points_transformed, rotation, center_transformed
        else:
            weights = self.ellipsoid(points, rotation_obs, translation_obs)
            weights = torch.softmax(weights, dim=-1)
            translation = translation_obs - self.ellipsoid_center_point
            # build rotation matrix
            original_rotation = general_utils.rotation_6d_to_matrix(self.ellipsoid_rotation)
            rotation = rotation_obs @ original_rotation.transpose(-1, -2)
            # build 4x4 transformation matrix
            transformation = torch.zeros((self.num_part, 4, 4), device='cuda')
            transformation[:, :3, :3] = rotation
            transformation[:, :3, 3] = translation
            transformation[:, 3, 3] = 1.0 # [num_part, 4, 4]
            # compute weighted transformation
            fwd = torch.matmul(weights, transformation.view(-1, 16)).view(-1, 4, 4) # [num_point, 4, 4]
            # build homogeneous coordinates for points
            homocord = torch.cat([points, torch.ones((points.size(0), 1), dtype=torch.float, device='cuda')], dim=-1)
            # transform points
            transformed_points = torch.matmul(fwd, homocord.unsqueeze(-1)).squeeze(-1)
            transformed_points = transformed_points[:, :3]
            # compute transformed center
            homocord_center = torch.cat([self.ellipsoid_center_point, torch.ones((self.num_part, 1), dtype=torch.float, device='cuda')], dim=-1)
            transformed_center = torch.matmul(transformation, homocord_center.unsqueeze(-1)).squeeze(-1)
            transformed_center = transformed_center[:, :3]
            return transformed_points, fwd, transformed_center

    def ellipsoid_volume_loss(self):
        # all ellipsoid should have comparable volume
        # compute the volume of each ellipsoid
        radii = torch.exp(self.ellipsoid_radii)
        a1 = radii[:, 0]
        a2 = radii[:, 1]
        a3 = radii[:, 2]
        volume = 4 * torch.pi * a1 * a2 * a3 / 3
        # compute the loss
        loss = 0.0
        for i in range(self.num_part):
            for j in range(i + 1, self.num_part):
                loss += torch.abs(volume[i] - volume[j])
        num_items = self.num_part * (self.num_part - 1) / 2.
        return loss / num_items



    def optimize(self):
        self.optimizer_rigid.step()
        self.scheduler.step()
        self.optimizer_rigid.zero_grad()
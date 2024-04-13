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
        points = self.ellipsoid_point_encodings(points)
        joints_embed = torch.cat([joints, points], dim=-1)

        rotation = joints_embed
        translation = joints_embed
        for i in range(self.num_layers):
            rotation = self.rotation_net[i](rotation)
            translation = self.translation_net[i](translation)
            if i < self.num_layers - 1:
                rotation = F.elu(rotation, alpha=0.1)
                translation = F.elu(translation, alpha=0.1)

        if self.rotation_representation == "quat":
            rotation = rotation.view(-1, 4)
            rotation = general_utils.build_rotation(rotation)
            rotation = rotation.view(3, 3)
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
        self.joints_embed = SimpleEmbedding(self.num_joints, self.embedding_dim)
        self.input_dim = self.embedding_dim
        self.decomposition = EmbeddingDecomposition(input_dim=self.input_dim, num_part=self.num_part)

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

    def forward(self, joints):
        # joints: [num_joints]
        # joints = self.encoding(joints)
        joints_embed = self.joints_embed(joints)
        joints_embed = self.decomposition(joints_embed)

        rotation = joints_embed
        translation = joints_embed
        for i in range(self.num_layers):
            rotation = self.rotation_net[i](rotation)
            translation = self.translation_net[i](translation)
            if i < self.num_layers - 1:
                rotation = F.elu(rotation, alpha=1)
                translation = F.elu(translation, alpha=1)

        if self.rotation_representation == "quat":
            # rotation = rotation.view(self.num_part, 4)
            rotation = general_utils.build_rotation(rotation)
        elif self.rotation_representation == "mat":
            # rotation = rotation.view(self.num_part, 6)
            rotation = general_utils.rotation_6d_to_matrix(rotation)

        # translation = translation.view(self.num_part, 3)
        # rotation: [num_part, 3, 3], translation: [num_part, 3]
        # convert to homogeneous transformation matrix [num_part, 4, 4]
        transformation = torch.zeros((self.num_part, 4, 4), device='cuda')
        transformation[:, :3, :3] = rotation
        transformation[:, :3, 3] = translation
        transformation[:, 3, 3] = 1.0
        return transformation



class GaussianDeformer(nn.Module):
    def __init__(self, num_joints, num_part=None, hidden_dim=128, num_layers=5, encoding=20, rotation_representation="quat", use_mlp=False, no_init=False):
        super(GaussianDeformer, self).__init__()
        self.rigid = RigidTransform_mul(num_joints, hidden_dim, 8, encoding, rotation_representation)
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
        # self.ellipsoid_radii = nn.Parameter(self.ellipsoid_radii, requires_grad=True)
        self.ellipsoid_rotation = torch.randn((self.num_part, 4), dtype=torch.float, device='cuda') # 6D rotation
        self.log_scale = nn.Parameter(torch.zeros(1, dtype=torch.float, device='cuda'))

        # register ellipsoid inited as buffer
        self.register_buffer("ellipsoid_inited", torch.tensor(no_init))
        self.use_mlp = use_mlp
        if use_mlp:
            self.weight_mlp = VanillaCondMLP(self.point_input_dim, 0, self.num_part)
        else:
            self.weight_mlp = VanillaCondMLP(self.point_input_dim + self.num_joints, 0, self.num_part)

        self.rest_pose = torch.ones(num_joints, dtype=torch.float, device='cuda') * 0.5

    def set_optimizer(self, max_steps=100000):
        self.optimizer_rigid = torch.optim.Adam(self.parameters(), lr=1e-4, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_rigid, max_steps, eta_min=1e-8)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_rigid, step_size=10000, gamma=0.5)
    def load(self, pa, optimizer=None, add=True):
        if add:
            self.ellipsoid_center_point = nn.Parameter(self.ellipsoid_center_point, requires_grad=True)
            if not self.use_mlp:
                self.ellipsoid_radii = nn.Parameter(self.ellipsoid_radii, requires_grad=True)
                self.ellipsoid_rotation = nn.Parameter(self.ellipsoid_rotation, requires_grad=True)
            if optimizer is not None:
                self.optimizer_rigid.add_param_group({"params": self.ellipsoid_center_point})
                if not self.use_mlp:
                    self.optimizer_rigid.add_param_group({"params": self.ellipsoid_radii})
                    self.optimizer_rigid.add_param_group({"params": self.ellipsoid_rotation})

        self.load_state_dict(pa)
        if optimizer is not None:
            self.optimizer_rigid.load_state_dict(optimizer)

    def get_canonical(self):
        # joints = torch.ones((self.num_joints,), dtype=torch.float, device='cuda') * 0.5 # joints are normalized to [0, 1], 0.5 is 0 degree
        # transformation = self.rigid(joints)
        rotation = general_utils.build_rotation(self.ellipsoid_rotation)
        translation = self.ellipsoid_center_point
        # rotation = transformation[:, :3, :3] @ rotation
        # translation = translation + transformation[:, :3, 3]
        return rotation, translation



    def set_ellipsoid(self, points, random=False):
        # if self.use_mlp:
        #     self.ellipsoid_inited = torch.tensor(True)
        #     return
        # use k-means to find the center of the ellipsoid
        if random:
            # random choose center from points
            center = points[torch.randint(0, points.size(0), (self.num_part,))]
            self.ellipsoid_center_point = nn.Parameter(center, requires_grad=True)
            if not self.use_mlp:
                self.ellipsoid_radii = nn.Parameter(torch.zeros((self.num_part, 3), dtype=torch.float, device='cuda'), requires_grad=True)
        else:
            kmeans = KMeans(n_clusters=self.num_part, random_state=0).fit(points.cpu().detach().numpy())
            ellipsoid_center_point = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device='cuda')
            self.ellipsoid_center_point = nn.Parameter(ellipsoid_center_point, requires_grad=True)
            # compute the radii of the ellipsoid
            if not self.use_mlp:
                distance = torch.cdist(points.detach(), ellipsoid_center_point, 2)
                distance = distance.cpu().numpy()
                # radii is the maximum distance from the center to the points belong to the ellipsoid
                radii = []
                for i in range(self.num_part):
                    radii.append(distance[kmeans.labels_ == i][..., i].max())
                ellipsoid_radii = torch.tensor(radii, dtype=torch.float, device='cuda')

                # expand to [num_part, 3]
                ellipsoid_radii = ellipsoid_radii.unsqueeze(-1).expand(-1, 3)

                self.ellipsoid_radii = nn.Parameter(torch.log(ellipsoid_radii), requires_grad=True)
        if not self.use_mlp:
            self.ellipsoid_rotation = torch.randn((self.num_part, 4), dtype=torch.float, device='cuda')
            self.ellipsoid_rotation[:, 0] = 1.0
            self.ellipsoid_rotation = nn.Parameter(self.ellipsoid_rotation, requires_grad=True)
        # add to optimizer
        self.optimizer_rigid.add_param_group({"params": self.ellipsoid_center_point})
        if not self.use_mlp:
            self.optimizer_rigid.add_param_group({"params": self.ellipsoid_radii})
            self.optimizer_rigid.add_param_group({"params": self.ellipsoid_rotation})
        self.ellipsoid_inited = torch.tensor(True)

    def ellipsoid(self, points, rotation, translation):
        # points: [num_point, 3]
        # compute Mahalanobis distance from each point to the surface of each ellipsoid
        # [num_point, num_part]

        # build rotation matrix
        # rotation = general_utils.build_rotation(self.ellipsoid_rotation)
        # rotation: [num_part, 3, 3]
        # build radii matrix [num_part, 3, 3]
        radii = torch.exp(self.ellipsoid_radii)
        L = torch.zeros((radii.size(0), 3, 3), dtype=torch.float, device="cuda")
        L[:, 0, 0] = radii[:, 0]
        L[:, 1, 1] = radii[:, 1]
        L[:, 2, 2] = radii[:, 2]
        L = L.unsqueeze(0)
        rotation = rotation.unsqueeze(0)

        # compute Mahalanobis distance as weights
        cord = points.unsqueeze(1) - translation.unsqueeze(0) # [num_point, num_part, 3]
        cord = cord.unsqueeze(-1) # [num_point, num_part, 3, 1]
        dis = cord.transpose(-1, -2) @ rotation.transpose(-1, -2) @ L @ rotation @ cord
        # small distance has large weight
        dis = dis.squeeze(-1).squeeze(-1)
        W = dis
        return W
        # dist = torch.cdist(points, translation, 2)
        # W = torch.exp(-dist ** 2 / (2 * torch.exp(self.ellipsoid_radii) ** 2))
        # return W


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

    def forward(self, points, joints, delta_mlp=True):
        # transformation = []
        # for i in range(self.num_part):
        #     transformation.append(self.rigid(joints, self.ellipsoid_center_point[i]))
        # transformation = torch.stack(transformation, dim=0)
        rotation_can, translation_can = self.get_canonical()
        transformation = self.rigid(joints)
        rotation = transformation[:, :3, :3]
        translation = transformation[:, :3, 3]
        # transformation: [num_part, 4, 4]


        if self.use_mlp:
            weights = self.weight_mlp(self.point_embedding(points))
            weights = torch.softmax(weights, dim=-1)

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

            return points_transformed, rotation, center_transformed, None
        else:
            translation_obs = translation_can + translation
            rotation_obs = rotation @ rotation_can
            weights = self.ellipsoid(points, rotation_can, translation_can)
            if delta_mlp:
                f = torch.cat([self.point_embedding(points), self.rest_pose.unsqueeze(0).expand(points.size(0), -1)], dim=-1)
                add_weights = self.weight_mlp(f)
                weights = weights + add_weights
            weights = torch.softmax(-1000 * self.log_scale.exp() * weights, dim=-1)

            # build 4x4 transformation matrix
            transformation = torch.zeros((self.num_part, 4, 4), device='cuda')
            transformation[:, :3, :3] = rotation
            transformation[:, :3, 3] = translation
            transformation[:, 3, 3] = 1.0 # [num_part, 4, 4]
            fwd = torch.matmul(weights, transformation.view(-1, 16)).view(-1, 4, 4)

            rotation_inv = rotation.inverse()
            translation_inv = -translation


            local_points = points.unsqueeze(1) - translation_can.unsqueeze(0)
            local_points_rotated = rotation @ local_points.unsqueeze(-1)
            transformed_points = local_points_rotated.squeeze(-1) + translation.unsqueeze(0) + translation_can.unsqueeze(0)
            transformed_points = torch.sum(transformed_points * weights.unsqueeze(-1), dim=1)


            # compute inverse transformation
            weights_inverse = self.ellipsoid(transformed_points, rotation_obs, translation_obs)
            if delta_mlp:
                f = torch.cat([self.point_embedding(transformed_points), joints.unsqueeze(0).expand(points.size(0), -1)], dim=-1)
                add_weights_inverse = self.weight_mlp(f)
                weights_inverse = weights_inverse + add_weights_inverse
            weights_inverse = torch.softmax(-1000 * self.log_scale.exp() * weights_inverse, dim=-1)
            inv_local_points = transformed_points.unsqueeze(1) - translation_obs.unsqueeze(0)
            inv_local_points_rotated = rotation_inv @ inv_local_points.unsqueeze(-1)
            inverse_points = inv_local_points_rotated.squeeze(-1) + translation_inv.unsqueeze(0) + translation_obs.unsqueeze(0)
            inverse_points = torch.sum(inverse_points * weights_inverse.unsqueeze(-1), dim=1)
            # compute transformed center
            # l2 loss
            cycle_loss = F.mse_loss(inverse_points, points)
            return transformed_points, fwd, translation_obs, cycle_loss


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
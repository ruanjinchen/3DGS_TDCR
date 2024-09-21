#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from pytorch3d.loss import chamfer_distance
import torch
from random import randint
import random

import torchvision

from utils.loss_utils import l1_loss, ssim, aiap_loss, center_loss, sparse_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from F_kinematic import GaussianDeformer

from utils.camera_utils import camera_from_camInfo
from lpipsPyTorch import lpips

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, train_until, use_mlp=False):
    # torch.autograd.set_detect_anomaly(True)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    dim_features = gaussians.get_features.shape[1] * 3
    transform = GaussianDeformer(dataset.joints, use_mlp=use_mlp)
    transform.set_optimizer(opt.iterations)
    transform.cuda()
    transform.train()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, transform_params, transform_opt_params, transform_sch_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        transform.load(transform_params, transform_opt_params, first_iter > train_until > 0 or train_until == 0)
        # transform.optimizer_pt.load_state_dict(transform_pt_opt_params)
        transform.scheduler.load_state_dict(transform_sch_params)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    total_pictrues = len(viewpoint_stack)
    # create a list to store the loss of each picture, we choose pictures randomly, pictures with higher loss will be selected more frequently
    loss_list = []
    for i in range(total_pictrues):
        loss_list.append(10.)
    ema_loss_for_log = 0.0
    ema_loss_for_log_image_only = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss_sum = 0.0
    scaler = torch.cuda.amp.GradScaler()
    for iteration in range(first_iter, opt.iterations + 1):

        if iteration > train_until > 0:
            if not transform.ellipsoid_inited:
                # use Kmeans to initialize the ellipsoid center
                transform.set_ellipsoid(gaussians.get_xyz, False)
        elif train_until == 0:
            if not transform.ellipsoid_inited:
                transform.set_ellipsoid(gaussians.get_xyz, True)


        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["renders"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        image_loss = 0.0
        total_loss = 0.0

        iter_start.record()


        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()



        for i in range(opt.accumulate_grad):

        # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()

            # Select a viewpoint based on the loss
            index = random.choices(range(len(viewpoint_stack)), k=1)[0]

            viewpoint = viewpoint_stack.pop(index)
            viewpoint_cam = camera_from_camInfo(viewpoint, 1.0, dataset)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            if iteration <= train_until and train_until > 0:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, deformer=None, render_mask=opt.lambda_mask > 0.0, render_bone=False)
            else:
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, deformer=transform, render_mask=opt.lambda_mask > 0.0, add_mlp=True, render_bone=True)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["renders"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            # show gt_image
            torchvision.utils.save_image(gt_image, "gt_image.png")
            Ll1 = l1_loss(image, gt_image)
            loss = opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + (1.0 - opt.lambda_dssim) * Ll1
            # loss += 0.5 * lpips(image, gt_image, net_type='vgg').squeeze()
            image_loss = loss.item()
            loss_list[index] = image_loss

            # loss = Ll1 + ssim(image, gt_image)
            # use mse loss for mask
            if opt.lambda_mask > 0.0:
                mask = render_pkg["mask"].squeeze(0)
                bone = render_pkg["bone"].squeeze(0) if render_pkg["bone"] is not None else None
                gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
                mask_loss = torch.nn.functional.l1_loss(mask, gt_mask)
                loss += opt.lambda_mask * mask_loss
                if bone is not None:
                    # # transform bone and mask to 2d point cloud
                    # image_size = torch.tensor([image.shape[1], image.shape[2]], device=image.device)
                    # # set bone to 0-1
                    # bone[bone < 0.5] = 0
                    # torchvision.utils.save_image(bone, "bone.png")
                    # bone = torch.nonzero(bone).float().unsqueeze(0)
                    # gt_mask = torch.nonzero(gt_mask).float().unsqueeze(0)
                    # # scale to 0-1
                    # bone = bone / image_size
                    # gt_mask = gt_mask / image_size

                    bone_loss = torch.nn.functional.l1_loss(bone, gt_mask)
                    loss += .1 * bone_loss
                    loss += .001 * transform.ellipsoid_volume_loss()
            if opt.lambda_aiap_xyz > 0.0 and iteration > train_until > 0:
                xyz_can, xyz_obs, cov_can, cov_obs, rotation_can, rotation_obs = render_pkg["xyz_can"], render_pkg["xyz_obs"], render_pkg["cov_can"], render_pkg["cov_obs"], render_pkg["rotation_can"], render_pkg["rotation_obs"]
                weights = render_pkg["weights"]
                aiap_loss_xyz, aiap_loss_cov, rigid_loss, rot_loss, smooth_loss = aiap_loss(xyz_can, xyz_obs, cov_can, cov_obs, rotation_can, rotation_obs, weights)
                loss += opt.lambda_aiap_xyz * aiap_loss_xyz + 0.1 * rigid_loss + 0.1 * rot_loss + 10 * smooth_loss
                # loss += 0.1 * sparse_loss(weights)
                # center_can = render_pkg["center_can"]
                center_obs = render_pkg["center_obs"]
                center_loss_v = center_loss(center_obs)
                loss += 0.1 * center_loss_v
                cycle_loss = render_pkg["cycle_loss"]
                if cycle_loss is not None:
                    loss += 1 * cycle_loss
                # rotations = render_pkg["rotations"]
                # loss += rig_loss(rotations)
            loss = loss / opt.accumulate_grad
            # scaler.scale(loss).backward()
            loss.backward()

        iter_end.record()



        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() * opt.accumulate_grad + 0.6 * ema_loss_for_log
            ema_loss_for_log_image_only = 0.4 * image_loss + 0.6 * ema_loss_for_log_image_only
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Image Loss": f"{ema_loss_for_log_image_only:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), deformer=transform, train_until=train_until)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
        if iteration < opt.iterations :
            if train_until == 0 or iteration > train_until:
                # scaler.step(transform.optimizer_rigid)
                transform.optimizer_rigid.step()
                transform.scheduler.step()
                # print(transform.scheduler.T_max)
                transform.optimizer_rigid.zero_grad(set_to_none = True)
                # gaussians.optimizer.step()
                # scaler.step(gaussians.optimizer)
            else:
                gaussians.optimizer.step()
                # scaler.step(gaussians.optimizer)
            gaussians.optimizer.zero_grad(set_to_none = True)
            # scaler.update()


        # if (iteration in checkpoint_iterations):
        if iteration >= train_until and iteration % 1000 == 0:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            print("current point number: ", gaussians.get_xyz.shape[0])
            torch.save((gaussians.capture(),
                        transform.state_dict(),
                        transform.optimizer_rigid.state_dict(),
                        # transform.optimizer_pt.state_dict(),
                        transform.scheduler.state_dict(),
                        iteration), scene.model_path + "/chkpnt_" + str(iteration) + ".pth")
            if iteration >= train_until + 1000 and not iteration - 1000 in checkpoint_iterations:
                if os.path.exists(scene.model_path + "/chkpnt_" + str(iteration - 1000) + ".pth"):
                    os.remove(scene.model_path + "/chkpnt_" + str(iteration - 1000) + ".pth")

        if iteration == 7000 or iteration == 300_000 or iteration == 150000 or iteration == 30000:
            break


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, deformer=None, train_until=7000):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        deformer.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint = camera_from_camInfo(viewpoint, 1.0, scene.args)
                    if iteration <= train_until:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, deformer=None)["renders"], 0.0, 1.0)
                    else:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, deformer=deformer, add_mlp=True)["renders"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/renders".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 12_000, 30_000, 60_000, 90_000, 120_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 550_000, 600_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 12_000, 30_000, 60_000, 90_000, 120_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 550_000, 600_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 12_000, 30_000, 60_000, 90_000, 120_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 550_000, 600_000])
    parser.add_argument("--start_checkpoint", "-k", type=str, default = None)
    parser.add_argument("--train_gaussian_until_iter", "-u", type=int, default=7000)
    parser.add_argument("--mlp", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.train_gaussian_until_iter, args.mlp)

    # All done
    print("\nTraining complete.")

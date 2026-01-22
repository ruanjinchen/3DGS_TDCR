'''
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 2 no base----------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=0
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.zero \
  -m out_tdcr2_no_base \
  --joints 6 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7001

检查第一阶段效果
export CUDA_VISIBLE_DEVICES=5
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.zero \
  -m out_tdcr2_no_base \
  --iteration 7000

  
cp out_tdcr2_no_base/chkpnt_7000.pth out_tdcr2_no_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=4
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.all \
  -m out_tdcr2_no_base_stage2 \
  -k out_tdcr2_no_base/chkpnt_7000_stage1_backup.pth \
  --joints 6 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7002
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_no_base.all \
  -m out_tdcr2_no_base_stage2 \
  --iteration 30000 \
  --eval \
  --skip_train


训练完成
  
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 2 with base--------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=1
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_with_base.zero \
  -m out_tdcr2_with_base \
  --joints 6 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7003


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=5
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_with_base.zero \
  -m out_tdcr2_with_base \
  --iteration 7000
  
cp out_tdcr2_with_base/chkpnt_7000.pth out_tdcr2_with_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=5
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_with_base.all \
  -m out_tdcr2_with_base_stage2 \
  -k out_tdcr2_with_base/chkpnt_7000_stage1_backup.pth \
  --joints 6 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7004
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=5
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/2m_with_base.all \
  -m out_tdcr2_with_base_stage2 \
  --iteration 30000

训练完成

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 3 no base----------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=4
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_no_base.zero \
  -m out_tdcr3_no_base \
  --joints 9 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7005


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_no_base.zero \
  -m out_tdcr3_no_base \
  --iteration 7000
  
cp out_tdcr3_no_base/chkpnt_7000.pth out_tdcr3_no_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=5
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_no_base.all \
  -m out_tdcr3_no_base_stage2 \
  -k out_tdcr3_no_base/chkpnt_7000_stage1_backup.pth \
  --joints 9 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7006
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_no_base.all \
  -m out_tdcr3_no_base_stage2 \
  --iteration 30000


训练完成
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 3 with base--------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=5
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_with_base.zero \
  -m out_tdcr3_with_base \
  --joints 9 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7007


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_with_base.zero \
  -m out_tdcr3_with_base \
  --iteration 7000
  
cp out_tdcr3_with_base/chkpnt_7000.pth out_tdcr3_with_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=3
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_with_base.all \
  -m out_tdcr3_with_base_stage2 \
  -k out_tdcr3_with_base/chkpnt_7000_stage1_backup.pth \
  --joints 9 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7008
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/3m_with_base.all \
  -m out_tdcr3_with_base_stage2 \
  --iteration 30000



-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 5 no base----------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=3
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_no_base.zero \
  -m out_tdcr5_no_base \
  --joints 15 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7009


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_no_base.zero \
  -m out_tdcr5_no_base \
  --iteration 7000
  
cp out_tdcr5_no_base/chkpnt_7000.pth out_tdcr5_no_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=4
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_no_base.all \
  -m out_tdcr5_no_base_stage2 \
  -k out_tdcr5_no_base/chkpnt_7000_stage1_backup.pth \
  --joints 15 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7010
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_no_base.all \
  -m out_tdcr5_no_base_stage2 \
  --iteration 30000



-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------SIM 5 with base--------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=0
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_with_base.zero \
  -m out_tdcr5_with_base \
  --joints 15 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7011


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_with_base.zero \
  -m out_tdcr5_with_base \
  --iteration 7000
  
cp out_tdcr5_with_base/chkpnt_7000.pth out_tdcr5_with_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=5
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_with_base.all \
  -m out_tdcr5_with_base_stage2 \
  -k out_tdcr5_with_base/chkpnt_7000_stage1_backup.pth \
  --joints 15 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7012
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/5m_with_base.all \
  -m out_tdcr5_with_base_stage2 \
  --iteration 30000






-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------REAL 2 with base--------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=3
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_2m_with_base.zero \
  -m out_real_tdcr2_with_base \
  --joints 6 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7003


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=5
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_2m_with_base.zero \
  -m out_real_tdcr2_with_base \
  --iteration 7000
  
cp out_real_tdcr2_with_base/chkpnt_7000.pth out_real_tdcr2_with_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=3
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_2m_with_base.all \
  -m out_real_tdcr2_with_base_stage2 \
  -k out_real_tdcr2_with_base/chkpnt_7000_stage1_backup.pth \
  --joints 6 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7004
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=5
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_2m_with_base.all \
  -m out_real_tdcr2_with_base_stage2 \
  --iteration 30000




-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------REAL 3 with base--------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

第一阶段训练 训练3D高斯 无形变
export CUDA_VISIBLE_DEVICES=0
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_3m_with_base.zero \
  -m out_real_tdcr3_with_base \
  --joints 9 \
  --lambda_mask 2.0 \
  --opacity_reset_interval 100000000 \
  -u 7000 --port 7007


检查第一阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_3m_with_base.zero \
  -m out_real_tdcr3_with_base \
  --iteration 7000
  
cp out_real_tdcr3_with_base/chkpnt_7000.pth out_real_tdcr3_with_base/chkpnt_7000_stage1_backup.pth

第二阶段 学习形变
export CUDA_VISIBLE_DEVICES=1
python train2.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_3m_with_base.all \
  -m out_real_tdcr3_with_base_stage2 \
  -k out_real_tdcr3_with_base/chkpnt_7000_stage1_backup.pth \
  --joints 9 \
  --lambda_mask 2.0 \
  --lambda_dssim 0 \
  -u 7000 \
  --iterations 30000 --port 7008
 
检查第二阶段效果
export CUDA_VISIBLE_DEVICES=0
python render.py \
  -s /data/yxk/K-data/K/fllm-sm/sim/3dgs/real_3m_with_base.all \
  -m out_real_tdcr3_with_base_stage2 \
  --iteration 30000

'''
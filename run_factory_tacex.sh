# """
#    Please activate the conda environment first by using "conda activate env_isaaclab"
#    This script is similar to run_factory.sh. The only difference is that it uses the TacEx-Factory-PegInsert-Direct-v0 environment, which includes tactile sensors, i.e. gelsight.
# """

## 1. Peg-in-Hole-Plus
#### Training
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Forge-PegInsert-Direct-v0 --headless --num_envs 1024 --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth

#### Data Collection
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth --headless 

#### Used for test
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge-peg-in-hole-plus-finetune-0413/test/nn/Factory.pth --headless 

## 1. Peg-in-hole-square
#### Training
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Forge-PegSquareInsert-Direct-v0 --headless --num_envs 1024 --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth

#### Data Collection
 ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegSquareInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_peg_in_hole_rectangle/test/nn/Factory.pth --headless 

## 2. Gear Assemble
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-GearMesh-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_gear/test/nn/Factory.pth --headless 

## 3. Gear Assemble plus
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-GearAssembly-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_gear/test/nn/Factory.pth --headless 

## 4. Nut
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-NutThread-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_nut/test/nn/Factory.pth  --headless 

## 1. rl training
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Factory-PegInsert-Direct-v0 --headless --num_envs 1024
 
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-GearMesh-Direct-v0 --headless  --num_envs 1024


# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-Factory-NutThread-Direct-v0 --headless  --num_envs 512

# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-AutoMate-Assembly-Direct-v0 --headless  --num_envs 512 


############################################################# Forge Env ###################################################
## 1. Peg-in-Hole
# (1) Training
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Forge-PegInsert-Direct-v0 --headless --num_envs 1024
# (2) Data Collection and Test
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth --headless 

# RL agent 继续训练
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task TacEx-Forge-PegInsert-Direct-v0 \
  --headless \
  --num_envs 1024 \
  --init-checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth


## 2. Gear
# (1) Training
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Forge-GearMesh-Direct-v0 --headless --num_envs 1024
# (2) Data Collection and Test
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-GearMesh-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_gear/test/nn/Factory.pth --headless 

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-RealSim-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth --headless 

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task TacEx-RealSim-PegInsert-Direct-v0 --headless  --num_envs 1024
 


./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Factory/test/nn/Factory.pth --headless --livestream 2


## 2. Data Collection using rl agent

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task 	TacEx-Forge-PegInsert-Direct-v0 --headless --num_envs 1024 --checkpoint logs/rl_games/Forge_peg_round_8mm/test/nn/Factory.pth

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge-peg-in-hole-plus/test/nn/Factory.pth --headless 

# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Factory_gear/test/nn/last_Factory_ep_200_rew_1490.1345.pth --headless 

# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-NutThread-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_nut/test/nn/Factory.pth

# nut 正确的
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Forge-NutThread-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Forge_nut/test/nn/Factory.pth  --headless 

# replay 轨迹
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/replay.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 8 --enable_cameras --checkpoint logs/rl_games/Factory/best/nn/Factory.pth --headless --hdf5_path dataset/episode_1.hdf5


## 3. Policy inference 
# pi0
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-pi0-v0 --num_envs 1 --enable_cameras --checkpoint /home/zhuchengyang/isaac-force-manip/logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew__281.6778_.pth --headless   --headless 


# tavla
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-tavla-v0 --num_envs 1 --enable_cameras --checkpoint /home/zhuchengyang/isaac-force-manip/logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew__281.6778_.pth --headless   --headless 


# pi0remote

PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task TacEx-Forge-GearMesh-Direct-v0 --num_envs 1 --enable_cameras \
  --checkpoint logs/rl_games/Forge_gear/test/nn/last_Factory_ep_200_rew__1527.7698_.pth --headless \
  --output_dir ./gear_baseline \
  --headless 2>&1 | tee logs/zcy/gear_baseline_$(date +%Y%m%d_%H%M%S).log


PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task TacEx-Forge-GearMesh-Direct-v0 --num_envs 1 --enable_cameras \
  --checkpoint logs/rl_games/Forge_gear/test/nn/last_Factory_ep_200_rew__1527.7698_.pth --headless \
  --output_dir ./gear_vae \
  --headless 2>&1 | tee logs/zcy/gear_vae_$(date +%Y%m%d_%H%M%S).log

# PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
#   --task TacEx-Factory-GearMesh-Direct-v0 --num_envs 1 --enable_cameras \
#   --checkpoint logs/rl_games/Factory_gear/test/nn/last_Factory_ep_200_rew_1490.1345.pth \
#   --output_dir ./test_gear \
#   --headless 2>&1 | tee logs/play_$(date +%Y%m%d_%H%M%S).log

# PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
#   --task TacEx-Factory-NutThread-Direct-v0 --num_envs 1 --enable_cameras \
#   --checkpoint logs/rl_games/Factory_NutThread/test/nn/last_Factory_ep_200_rew_933.81506.pth \
#   --output_dir ./test_nutthread \
#   --headless 2>&1 | tee logs/play_$(date +%Y%m%d_%H%M%S).log




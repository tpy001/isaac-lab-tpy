"""
    This script is used to run the reinforcement learning training for different tasks (Peg-in-hole, gear insertion, etc.). This is developed by original NVIDIA Isaac Lab.
    More information can be found in 'https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html'
"""

# Training
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-GearMesh-Direct-v0 --headless   --num_envs 1024

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-AutoMate-Assembly-Direct-v0 --headless   --num_envs 1024

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-AutoMate-Assembly-Direct-v0 --headless  --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Assembly_00015/test/nn/last_Assembly_ep_600_rew_636.90765.pth --headless --livestream 2

./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task TacEx-Factory-PegInsert-Direct-v0 --num_envs 1 --enable_cameras --checkpoint logs/rl_games/Factory_peg/test/nn/last_Factory_ep_200_rew_769.0327.pth --headless --livestream 2

# Test
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 1  --checkpoint /home/zhuchengyang/isaac-force-manip/logs/rl_games/Factory/test/nn/Factory.pth  --headless


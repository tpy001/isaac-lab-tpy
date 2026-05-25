## 建议多测试几次，IsaacLab仿真的随机性非常大，加上Flow Match 算法本身具有不确定性，因此需要多测算几次
## Peg-in-hole-plus

| 方法 | 执行步数 | 成功次数 | 成功率 |
| -- | -- |  -- | -- |
| RL agent  | \ | 188/200 | 94.0% |   
| pi0 (LoRA)      | 32 |  | 36/100 = 36%  | episode_length_s = 20  25000
| pi0 (LoRA)      | 32 |  | 46/100 = 46%  | episode_length_s = 20  29999
| pi0 (Full)      | 32 |  | 43/100 = 43%  | episode_length_s = 20
| TA-VLA (LoRA)  | 45 / 100 = 45% | episode_length_s = 20
| MOT  (LoRA) | 30 / 100 = 30% | episode_length_s = 20 ckpt 20000
| MOT_new  (LoRA) | ( 64 + 58 ) / 100 = 61% | episode_length_s = 20 ckpt 20000
| MOT  (LoRA) | 56 / 100 = 56% | episode_length_s = 20 ckpt 25000
| MOT_new  (LoRA) | 36 / 100 = 36% | episode_length_s = 20 ckpt 25000
| MOT  (LoRA) | 54 / 100 = 54% | episode_length_s = 20 ckpt 29999
| MOT_new  (LoRA) | 19 / 52 = 36% | episode_length_s = 20 ckpt 29999
| MOT  (Full) | 35 / 100 = 35% | episode_length_s = 20 ckpt 20000
| MOT  (Full) | 32 / 100 = 32% | episode_length_s = 20 ckpt 29999

## Gear 0225
| 方法 | 执行步数 | 成功次数 |  成功率 |
| -- | -- |  -- | -- |
| RL agent | \  | 199/200 | 99.5% |
| pi0      | 32 | 78/100 | 78% | 
| Tavla      | 32 | 60/100 | 60% |  
| MOT      | 32 | 34/100 | 34% |  






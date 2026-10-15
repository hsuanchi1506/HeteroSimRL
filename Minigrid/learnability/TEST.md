我有個有趣的想法：我們想做的東西就是用最有效率的方法，e.g. 用最少的算力讓模型練習，訓練出一個好用的 robot controller。

這東西很像是 Exploration，像是 AlphaNIST 用 Hetero Envs 搭配 SGS，讓 PPO 可以從 task 中間的 state 開始練習。Play2Perfect 則是先讓 policy 練習把物體移動、旋轉到不同的 Goal Pose，再去 fine-tune downstream tasks。

直接 train downstream task 不只不 general，更有可能它根本不是 train 出 optimal policy 的好方法；而先練某個簡單技能，再拿 pre-train 好的 policy 去 train downstream task，才可能是比較有效率的 training path。

換句話說，我覺得可以加速的地方就是在「練習」這件事，到底怎麼有效率的練習？

我用 MiniGrid 做了一些 Toy Examples，一些觀察：
1. 練習對的東西有用。
  * Task setup: Policy 要拿鑰匙、開門、走到終點，演算法我用 PPO；因為任務很簡單，所以模型架構就是共用 MLP features 接一個 policy head 跟 critic head。所有的 reward 都是 sparse reward。 
  * Pre-training 都是先做 2M steps。
  * 分成三種 case：train from scratch 是直接訓練拿鑰匙開門走到 goal， `open_door` 指的是先練習過開門再拿 policy, `face_door` 指的是先練習站在門前，而且面向門。
  * [影片 1：Practice can make an agent learn faster or slower](videos/1_practice_can_make_an_agent_learn_faster_or_slower.mp4)
  * 這邊最有趣的點是我發現 Pre-training 中途的 checkpoint，`open_door` 只練 50K steps，開門本身的 task success rate 只有 2%，但拿這個 checkpoint 直接去做 downstream task 就已經跟 final 很接近了。
  * 換句話說練習可能不用真的練很久，不需要把 Pretraining Task 練到 100%，也許練一下以後就可以讓他上場去賺經驗。

| Pretraining | 4M 步內每個 checkpoint 直接平均的成功率 (AUC) | 學會所需的 Steps |
|---|---:|---:|
| Scratch | 0.42 | 2.42M |
| open_door | 0.90 | 0.43M |
| face_door | 0.00 | 0.00 |

2. 為啥 `face_door` 那麼爛？答案是 Critic 在搞。

結果 `face_door` 的 AUC 從 0.00 變成 0.55。

後來讓 Claude Code 隨便測了 14 種 Pretraining Tasks，發現 Critic 重學都會比較好。

[影片 2：The old critic stops learning](videos/2_the_old_critic_stops_learning.mp4)

## 3. Policy 學的任務不能讓 model 做有矛盾的行為
我另外比較了兩個 tasks。
Task A： 拿鑰匙 → 開門 → 走到終點。

Task B： 拿鑰匙 → 開門 → 放下鑰匙 → 撿球。

結果 Task A 很快就學會了，但 Task B 從 Scratch 跑了 10M Steps 都學不會。

一開始我猜是 Task B 太難 Exploration，但我直接 deploy random Policy 成功完成 Task B 的機率竟然比 Task A 還高。
而且兩個 task 都在訓練前期都有出現過 success，而且越訓練越糟，Agent 越來越不願意碰物體，最後 Success Rate 直接歸 0。

[影片：Without practice, more training makes the agent touch fewer objects](videos/4_without_practice_more_training_makes_the_agent_touch_fewer_objects.mp4)

所以我又做了一個實驗，Task B 改成 Agent 手上直接有鑰匙：

(metrics 是達到 80% 成功率所需的 Training Steps。)
| 初始狀態 | Task A | Task B |
|---|---:|---:|
| 從頭開始 | 1.97M | 學不會 |
| 開局手上有鑰匙 | 0.59M | 3.76M |
| 開局手上門已經打開 | 0.29M | 0.09M |

所以我覺得 RL train controller 的重點是不能讓 model 做矛盾的行為，例如不能叫他先拿起來鑰匙又叫他放下來，這是 planning 要做的事。

總結就是我覺得兩個可以想的問題：
1. practice 要練什麼
2. 除了 pre-training task success rate 以外有沒有其他可以量的東西？像是上面我 `open_door` 練到 2% 其實就已經夠用了，顯然 success rate 不是一個好 metric。


metrics 是達到 80% 成功率所需的 Training Steps
| 初始狀態 | Task A | Task B |
|---|---:|---:|
| 從頭開始 | 1.97M | 學不會 |
| 開局手上有鑰匙 | 0.59M | 3.76M |
| 開局手上門已經打開 | 0.29M | 0.09M |
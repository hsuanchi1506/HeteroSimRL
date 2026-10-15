對，我也覺得。**MiniGrid 是不是自己寫的、在哪個環境重現，其實都不是這個故事的重點。** 我們只是用一個簡單的 Toy Example，想理解為什麼有些任務直接練很難學，但換一種練習方式就容易很多。

我把影片 3 和整段官方 MiniGrid 重現拿掉，也把一些 Implementation Details 刪掉，讓故事集中在最初的研究問題。

---

# What Makes a Task Learnable?

我最近一直在想一個 Robot Learning 的問題：

**為什麼機器人需要練習幾百萬次才能學會一件事？有沒有可能透過更聰明的練習，用更少的算力學會更多東西？**

這個想法其實是從 AlphaNIST 跟 Play2Perfect 來的。

AlphaNIST 用 Hetero Envs 搭配 Success-Guided Sampling，讓 PPO 可以從任務中間的狀態開始練習。Play2Perfect 則是先讓 Policy 練習把物體移動、旋轉到不同的 Goal Pose，再去 Finetune 精密組裝任務，結果比直接從頭練有效率很多。

我原本覺得，這兩個方法可能都是在改善 **Exploration**，讓 Policy 更容易探索到成功的 Trajectory。

但後來想一想，感覺核心可能不完全是這個。

假設今天要學鎖螺絲，直接用 PPO 練可能要幾百萬步，但如果先學會抓取、旋轉、調整物體姿態，後面就容易很多。

**有沒有可能這些練習真正改善的是 Learnability？也就是讓原本很難學會的東西，變得比較容易學。**

所以我很好奇這背後到底是什麼原因，做了一些簡單的 MiniGrid Toy Experiments。

## 1. 練對的東西，真的會比較快學會新任務嗎？

我先讓 Policy 練習不同的任務，再去 Finetune `goal`，也就是拿鑰匙、開門、走到終點。

結果蠻有趣的：

| 訓練方式 | Learning AUC | 學會所需 Steps |
|---|---:|---:|
| Scratch | 0.42 | 2.42M |
| **Pretrain `open_door`** | **0.90** | **0.43M** |
| Pretrain `face_door` | 0.00 | 4M 內學不會 |

練過 `open_door` 的 Policy，學習速度快了非常多（p = 0.0005）。但練過 `face_door` 的反而比 Scratch 還差（p = 0.005）。

[影片 1：Practice can make an agent learn faster or slower](videos/1_practice_can_make_an_agent_learn_faster_or_slower.mp4)

影片裡一開始三個 Agent 都沒辦法完成 `goal`。

但練過 `open_door` 的 Agent 已經會拿鑰匙開門，練過 `face_door` 的 Agent 則只會走到門前。

Finetune 0.5M Steps 後，`open_door` 已經學會了。3M Steps 後 Scratch 也學會了，但 `face_door` 還是不會。

**所以練對的東西真的有幫助，但練錯的東西甚至會讓後面的任務更難學。**

## 2. 為什麼練過 face_door 反而更難學？

這個我覺得更奇怪，所以又做了一些實驗。

原本 Pretraining 完會把整個 Actor-Critic 都搬去新任務繼續訓練。

但問題是，**Critic 學到的是舊任務的 Value Function，它對新任務的預期可能完全是錯的。**

例如練過 `face_door` 的 Critic，看到 Agent 站在門前就認為快拿到 Reward 了。但新任務是走到終點，光站在門前根本沒有用。

這些錯誤的 Value Estimates 可能透過 PPO 的 Advantage Estimation，干擾後續的 Policy Update。

所以我試了一個很簡單的方法：**Actor 全部保留，只把 Critic 最後一層的 257 個參數歸零，再繼續訓練。**

結果 `face_door` 的 AUC 中位數直接從 0.00 變成 0.55（p = 0.0016）。

後來我把 14 種 Pretraining Tasks 都測了一遍，發現 **14/14 重置 Critic 都比直接沿用整個網路好**，其中 11 種達到 p ≤ 0.01。

[影片 2：The old critic stops learning](videos/2_the_old_critic_stops_learning.mp4)

這個影片左右兩個 Agent 一開始的 Actor 完全一樣，差別只有 Critic。

左邊的 Critic 預測 Value 大約 0.91，右邊重置後是 0。

結果右邊很快就開始探索，2M Steps 後學會任務；左邊卻一直拿著鑰匙站在門前。

**所以有點像是 Pretraining 其實已經學到有用的能力，但舊的 Critic 反而讓 Policy 沒辦法有效利用這些能力去學新任務。**

## 3. 但我又發現，有些任務明明探索得到成功，PPO 還是學不起來

我另外比較了兩個任務：

**Task A (`goal`)：**
拿鑰匙 → 開門 → 走到終點

**Task B (`open_with_ball`)：**
拿鑰匙 → 開門 → 放下鑰匙 → 撿球

Task A 很快就學會了，但 Task B 從 Scratch 開始，4 個 Seeds 跑了 10M Steps，全部學不會。

一開始我以為是 Task B 太難探索到成功。

但實際測出來，隨機 Policy 完成 Task B 的機率反而比 Task A 高：

- Task A：6.3e-4
- Task B：9.9e-4

而且兩個任務在訓練前期都已經成功過。

**所以問題不是找不到成功，而是明明成功過，PPO 卻沒辦法學會怎麼穩定重現。**

更奇怪的是，Task B 越訓練越糟。Agent 越來越不願意碰物體，撿鑰匙的比例從 40% 掉到 23%，撿球從 42% 掉到 19%，最後連原本偶爾成功的機率都掉到接近 0。

[影片：Without practice, more training makes the agent touch fewer objects](videos/4_without_practice_more_training_makes_the_agent_touch_fewer_objects.mp4)

後來我做了一個實驗，直接讓環境幫 Agent 完成前面的步驟。

| 初始狀態 | Task A | Task B |
|---|---:|---:|
| 從頭開始 | 1.97M（4/4） | 學不會（0/4） |
| 開局手上有鑰匙 | 0.59M（6/6） | 3.76M（5/6） |
| **開局門已經打開** | **0.29M（6/6）** | **0.09M（6/6）** |

數字是達到 80% 成功率需要的 Steps，括號是學會的 Seeds 數。

[實驗圖：Residual Chain](figs/fig_residual.png)

**原本怎麼練都學不會的 Task B，只要開局門已經打開，90K Steps 就學會了。**

這讓我想到一個問題：會不會有些 Task Structure，本來就特別不適合直接用 PPO 學？

例如 Task A 每個步驟都在往最終目標前進。拿到鑰匙、打開門之後，就可以繼續走到終點。

但 Task B 必須先拿到鑰匙、打開門，接著又要主動放下鑰匙，才能去拿球。

也就是 **Policy 必須先學會一個有用的行為，然後又在正確的時間放棄它。**

我目前還不能確定換手是不是唯一的瓶頸，因為開局門已經打開也同時縮短了任務。不過這個現象讓我覺得，問題可能跟 Credit Assignment、Task Structure，甚至 Planning 有關。

而且我發現，對於那些能學會的任務，Pretraining 真正縮短的也不是找到第一次成功的時間。

如果把學習分成：

- **Discovery**：第一次探索到成功。
- **Consolidation**：從第一次成功，到穩定達到 80% 成功率。

Scratch 大約 0.2M Steps 就找到第一次成功了。

但是從第一次成功到真正學會，還需要約 1.1M Steps。

練過 `open_door`、`enter_right` 這些任務後，Consolidation 可以縮短到大約 0.25M Steps。

[實驗圖：Discovery vs. Consolidation](figs/fig_decomp.png)

所以我現在反而覺得，**練習最重要的價值，可能不是讓 Policy 更容易找到成功，而是讓 Policy 更容易從成功經驗中學到東西。**

這樣回頭看 AlphaNIST 跟 Play2Perfect，好像就比較能解釋為什麼有效。

AlphaNIST 直接從中途開始，相當於讓環境幫 Policy 完成前面一部分的行為鏈。

Play2Perfect 則是先讓 Policy 把一些基本操作練成習慣，後面 Finetuning 只需要學剩下的部分。

當然，這只是目前從 Toy Experiments 得到的解釋，還沒有直接在 Play2Perfect 上驗證。

## 4. 所以現在我在想什麼？

我覺得目前有三個蠻有趣的觀察。

**第一，Agent 可能還沒學會一件事，但已經變得更容易學會其他事情。**

所以 Pretraining 的成功率，不一定能衡量這段練習對未來到底有多少價值。

**第二，練習學到的 Skills 跟 Critic 的預期，可能要分開看。**

Skills 可以 Transfer，但舊 Critic 可能因為對新任務的 Value 預測不準，讓 PPO 產生有害的更新。這在目前的 PPO 設定下很明顯，但還不知道能不能推廣到其他 RL Algorithms。

**第三，探索到成功，跟真正學會成功，是兩個不同的問題。**

有些任務明明已經成功過，PPO 還是沒辦法把它學成穩定的行為。但只要先完成一部分任務，或者先練好相關能力，剩下的部分就突然變得很容易學。

這也讓我重新回到最開始的問題：

**What makes a task learnable?**

以及更進一步：

**What should an agent practice to become better at learning?**

我覺得這件事跟我們現在做的 High-Throughput Simulation 有很直接的關係。

我們現在一直在想怎麼增加 Environments、提高 Simulation Throughput，讓 PPO 可以用更短的時間跑更多 Steps。

但假設有些練習只需要很少的 Steps，就能讓後續任務的學習成本下降好幾倍，那我們是不是應該反過來想：

**給定一個 GPU Budget，到底該讓 Policy 練習什麼，才能讓它未來學會最多東西？**

而且這才真正開始跟 Self-Improvement 有關。

因為一個能夠自主學習的 Agent，理論上應該要知道自己目前缺少什麼能力、決定要練什麼、練到什麼程度，以及怎麼把以前學到的能力拿來學新東西。

目前還沒有測到新物體、新場景的 Generalization，也還不知道該怎麼讓 Agent 自己找到這些有效的練習。

但我覺得至少已經看到一個蠻值得繼續追的現象：

**同樣是幾百萬個 Environment Steps，不同的練習內容跟學習路徑，最後得到的能力可能差非常多。**

我現在最想知道的是，能不能找到一個比較 Fundamental 的原則，讓 Agent 在不知道未來任務的情況下，也能自己找到這些有效的練習路徑。
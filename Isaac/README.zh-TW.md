# Isaac / PhysX（中文版）

Isaac Lab 官方 Factory 三個任務（插銷、齒輪、螺帽）和 Play2Perfect 四個組裝任務的 Homo／Hetero 版本，照 `../RULES.md` 做。英文版：[README.md](README.md)。結果：[results/README.zh-TW.md](results/README.zh-TW.md)。

## 狀態（2026-10-10）

- **做完**：
  - 聯集場景的環境（Homo 和 Hetero 共用）、訓練、評估、梯度分析、錄影、計時和剖析的程式。
  - 128 個環境的 Hetero／Homo 訓練：官方 Hetero 3 個種子，另外兩組各 5 個種子。
  - RULES 第 6 節：用訓練好的策略，在 3、128、4096 個環境記接觸、階段、等待。
  - 第 7 節：讀 PhysX 原始碼。
  - 10/07 開會的 TODO：Homo 的 advantage 標準化、分散算力、更細的剖析。
  - RULES「每個實驗跑一次」的正式版（`rules_*`）：128、3、4096 個環境，含定期評估和影片。
- **Play2Perfect**（2026-10-13）：Homo／Hetero 訓練過（一個種子）。Hetero 跑完；Homo 在官方規模下 GPU 記憶體不夠、跑不完，當成結果。見 [Play2Perfect](#play2perfect進行中)。

## 環境

| | |
|---|---|
| 機器 | gputw1：RTX PRO 6000 Blackwell Workstation（96 GB），驅動 595.84，功率上限 500 W，跟別人共用 |
| 軟體 | Isaac Sim 5.1.0、Isaac Lab v2.3.2（原始碼）、PyTorch 2.7.0+cu128、rl_games 1.6.1（Isaac Lab 用的版本）、Python 3.11（conda） |
| 執行 | 在 IsaacLab 資料夾裡，HeteroSimRL 放在它旁邊：`./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/<腳本>.py --headless ...` |
| PhysX 設定（官方） | 物理步 1/120 秒；每個策略步 8 個物理步；位置迭代 192 次；GPU 暫存區用官方值，只有碰撞暫存區在 1024 個環境以上加大（見下面） |

注意：gputw1 的驅動（595）讓 Isaac Sim 5.1 的算圖器（RTX）一開就當掉，連空場景也一樣。不開攝影機的模擬和訓練不受影響，錄影改用「PhysX 記位置、MuJoCo 畫圖」（見「錄影」）。

## 做法：聯集場景

官方 Factory 一個環境只能跑一種任務。Hetero 要讓同一批環境同時跑三種任務：

- 每個環境都放齊 8 個零件：插銷和孔、中齒輪和齒輪底座、小齒輪、大齒輪、螺帽和螺栓。每個環境一次只跑一個任務，其他任務的零件停在機器人後方 0.6 m，機器人碰不到。
- 每個零件都是官方的 `ArticulationCfg`，只改 prim 路徑（兩個任務不能都叫 `/FixedAsset`）。
- 機器人、控制器、模擬設定、觀測雜訊都直接繼承官方的 `FactoryEnvCfg`。
- 每個任務不同的參數（起點、雜訊、獎勵係數、成功門檻……）從官方設定和函式讀成一張表，依每個環境的任務編號取值，不手抄數字（`TASK_PARAMS.md`）。
- 觀測多一個任務的 one-hot（策略 22 維、critic 46 維）。
- Homo 和 Hetero 用同一個場景，差別只在每個環境分到哪個任務。

驗證過的：Homo 模式、同一個亂數種子，三個任務的位置、觀測、獎勵都和官方 `FactoryEnv` 一樣（離線檢查）。聯集場景和「每個環境只放自己零件」的大 list 場景，每一步的位置完全一樣，所以停放的零件不影響物理。

為什麼不用 Isaac Lab 的 `MultiUsdFileCfg`：它要求每個檔案的 prim 名稱都一樣，官方零件的名稱不一樣。

## 檔案

| 檔案 | 內容 |
|---|---|
| `hetero_factory/hetero_factory_env_cfg.py` | 聯集場景：8 個零件槽、停放位置、`task_onehot`、任務分配設定 |
| `hetero_factory/hetero_factory_env.py` | 環境本體，繼承官方 `FactoryEnv`，只改寫跟任務有關的方法，每一處都標 `HETERO`。Gym id：`Isaac-HeteroFactory-Direct-v0` |
| `hetero_factory/hetero_factory_list_env.py` | 大 list 場景（每個環境只放自己任務的零件）。Gym id：`Isaac-HeteroFactoryList-Direct-v0` |
| `hetero_factory/task_table.py`、`TASK_PARAMS.md`、`print_task_table.py` | 每個任務的參數表、內容和出處、在訓練機器上重新產生來比對 |
| `hetero_factory/assignment.py` | 每個環境分到哪個任務：Hetero 平均分（多出來的每回合輪流換）、Homo 全部同一個；分析用的固定數量（`counts`） |
| `hetero_factory/train_common.py`、`train_common_args.py` | 兩個訓練腳本共用：官方 PPO 設定、RULES 允許的改動、紀錄 |
| `hetero_factory/rl_utils.py` | `env.step` 計時、GPU 使用率、每輪一行的紀錄、advantage 的按任務／合併標準化、Homo 換任務 |
| `hetero_factory/grad_probe.py` | 梯度分析（RULES 第 5 節） |
| `hetero_factory/profiling.py` | GPU 程式分組（接觸偵測、解算、關節……） |
| `hetero_factory/pose_record.py`、`video_utils.py` | 錄影用 |
| `hetero_factory/offline_check/` | 不需要 Isaac Lab 的離線檢查（替身模組），說明在裡面的 README |
| `scripts/train_hetero.py`、`train_homo.py` | 訓練，照 Isaac Lab 官方 rl_games `train.py` 改 |
| `scripts/eval_policy.py` | 評估：每個任務單獨考、關掉隨機性 |
| `scripts/analyze_runs.py`、`plot_groups.py` | 成功率、獎勵對步數和時間的表和圖；多個種子畫成平均和範圍；梯度圖 |
| `scripts/record_video.py`、`render_offline.py` | 錄影：PhysX 記位置，MuJoCo 畫 |
| `scripts/bench_scene.py` | 亂數動作計時：聯集 vs 大 list、每個階段的時間、PhysX 統計 |
| `scripts/bench_policy.py` | 訓練好的策略計時（RULES 第 6 節）：整個回合、接觸點數和階段、GPU 剖析視窗、固定任務數量、多程序同時起跑 |
| `scripts/profile_step.py` | GPU 程式剖析（PyTorch 剖析器；也可以只開 Nsight 的錄製範圍） |
| `scripts/check_active_pose.py` | 檢查「一次讀完」和「一個任務讀一次」讀到的零件位置一樣 |
| `scripts/smoke_env.py` | 第一次在新機器上跑的煙霧測試 |

## 怎麼跑

```bash
# 訓練（官方 PPO、200 輪）
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_hetero.py --num_envs 128 --seed 0 --headless
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_homo.py --num_envs 128 --seed 0 --switch rollout --block 4 --headless
# 評估（給 nn/ 資料夾會挑最後的存檔）
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/eval_policy.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --out runs/eval/hetero_n128_s0.json
# 圖（不需要 Isaac Sim）
python ../HeteroSimRL/Isaac/scripts/plot_groups.py --out runs/analysis \
    --group "Hetero" tab:blue logs/rl_games/Factory/hetero_n128_s0 --group "Homo" tab:orange logs/rl_games/Factory/homo_rollout4_n128_s0
# 用訓練好的策略計時（第 6 節）
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/bench_policy.py --headless --num_envs 128 \
    --checkpoint logs/rl_games/Factory/hetero_n128_s1_ptadv/nn --contacts --profile_windows 5,300 --out runs/sec6/n128_hetero.json
```

訓練常用選項：

- `--num_envs`：3、128、4096。
- `--seed`。
- `--adv_norm global|per_task|pooled`：advantage 的標準化方式，見下面的說明。
- `--save_every N`：每 N 輪存檔，給定期評估和錄影用。官方是 100，這個只改存檔頻率，不是 PPO 設定。
- `--collision_stack_size`：碰撞暫存區大小（位元組）。
- `--grad_every`：梯度分析的頻率。

輸出在 `logs/rl_games/Factory/<run>/`：

- `run_info.json`：設定、對官方的改動、版本。
- `metrics.jsonl`：每輪一行，有時間、`env.step` 時間、GPU、每個任務的樣本數、advantage 統計。
- `episodes.jsonl`：每個回合、每個任務一行。
- `grad_diag.jsonl`：梯度分析。

## 對官方設定的改動

RULES 允許或需要的：

| 改動 | 原因 |
|---|---|
| 3 個環境時 minibatch 512 → 384 | 一輪只有 3×128 = 384 筆，不夠切 512（RULES：可以為了整除改） |
| 1024 個環境以上，碰撞暫存區加大（Hetero 1 GB、Homo 2 GB） | 官方的 2^28 只夠 128 個環境，不夠會丟接觸。這是模擬器的緩衝區，不是任務或 PPO 設定 |
| 回合統一 30 秒 | 官方是插銷 10、齒輪 20、螺帽 30 秒，但官方的重設要所有環境一起做；暫定用最長的，等組裡決定 |
| `save_frequency` 改成 20（只有正式版） | 第 8 節要定期評估和錄影 |

多做的分析（都有開關，不開就是官方設定）：

| 項目 | 為什麼做 |
|---|---|
| `--adv_norm per_task` | 官方把一整批的 advantage 一起標準化，Hetero 的一批裡混了三個任務，獎勵最小的螺帽整體被判成負的，有的種子學不會。改成每個任務各自標準化後，三個任務都學會。這是用來證明原因的實驗 |
| `--adv_norm pooled` | 10/07 開會：Homo 每一輪只有一個任務，官方的標準化對它來說等於按任務標準化。這個選項讓 Homo 用「三個任務最近各一批」合起來的統計量，對應官方 Hetero 的做法；對 Hetero 來說就是官方的做法 |
| 讀零件位置「一次讀完」（`fast_active_pose`，預設開） | 原本每個物理步一個任務讀一次，Hetero 要讀三次，佔每步約 2%。改成一次讀完，讀到的值完全一樣（`check_active_pose.py`） |
| `bench_policy.py`、`profile_step.py`、Nsight | 第 6 節要量每個任務的步時間、接觸、等待；亂數動作不會真的把零件裝上去，會低估混在一起的代價，所以用訓練好的策略 |
| `--task_counts`（只在計時用） | 10/07 開會：每個任務給不同數量的環境，看能不能把等待的時間填滿、或照難度分組 |

## 大 list 場景（`--scene list`）

每個環境只放自己任務的零件，所有環境仍然在同一個 PhysX 場景裡。

- 用來量聯集場景「多放零件」要付多少代價：
  - 3、128 個環境時差 2%；
  - 1024、4096 個環境時，聯集慢約 1.8 倍。原因是停放的齒輪一直坐在停放的底座上，接觸一直要算。
- 每個環境獨立建立（`replicate_physics=False`），因為複製第 0 號環境會把它的零件放進所有環境。
- 每個零件只生成在它任務的環境裡，讀寫透過「全域環境編號 → 資料列」的對照表。
- 只能跑 Hetero，任務在建場景時就固定。

## Advantage 的標準化

rl_games 會把一整批資料的 advantage（這個動作比平均好多少）一起調成平均 0、標準差 1。

- `per_task`：在 rl_games 建好資料、開始更新之前（`after_steps`），用觀測裡的任務 one-hot 把資料分開，各自標準化。
- `pooled`：在 rl_games 標準化之前，先記下原本的 advantage；每個任務保留最近一批的平均和變異數，再用所有任務合起來的平均和標準差標準化。
- 兩種都不改 rl_games 本身。不管開不開，`metrics.jsonl` 都會記每個任務的 `adv_mean_<任務>`、`adv_std_<任務>`。

## Homo 換任務

rl_games 每 128 步更新一次，回合是 449 步，兩者對不齊，所以「每次更新只有一種任務」和「回合不切斷」不能同時成立。

- `--switch rollout`（預設，照 RULES 的字面）：每 `--block` 次更新換一次任務。最後一步讓所有環境同時時間到，被切斷的回合用 critic 補上剩下的價值，紀錄標成 `_truncated`。
- `--switch episode`：每 `--block` 個完整回合換一次，跨過換任務那次的更新會混到兩個任務。

不管哪一種，一個循環（插銷、齒輪、螺帽各一段）後，Homo 和 Hetero 每個任務的樣本數都一樣（`metrics.jsonl` 的 `samples_per_task`）。

## 梯度分析（RULES 第 5 節）

在每次 PPO 更新開始前、用更新前的參數量：

- 每個任務的 actor 梯度大小；
- 任務之間的梯度夾角；
- 每個任務跟整體的夾角；
- 連續兩次更新的夾角。第 1、2 輪量，之後每 `--grad_every` 輪量一對相鄰的更新；
- 每一輪參數實際的變化。

只觀察、不影響訓練：不動 `.grad`、optimizer、資料和全域亂數。量的時間另外記，`wall_time_excl_diag` 是扣掉之後的時間。

## 錄影

```bash
# 1. 跑物理、記下每一步每個剛體的位置（不開攝影機）
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/record_video.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --seconds 30 --out runs/video/hetero_n128_s0
# 2. 用 MuJoCo 畫（另一個小環境：numpy、mujoco、imageio-ffmpeg、pillow）
MUJOCO_GL=egl python ../HeteroSimRL/Isaac/scripts/render_offline.py runs/video/hetero_n128_s0.npz
```

- 一列一個任務，每列 3 格（每格 240×180），達到成功條件時邊框變綠。
- 驅動正常的機器可以用 `--renderer rtx --enable_cameras` 直接用 Isaac Sim 算圖。
- 計時和剖析的時候不錄影。

## Play2Perfect（進行中）

Play2Perfect 四個組裝任務（緊配合插入、橫樑組裝第一步和第二步、鎖螺絲）的 Homo／Hetero 版本，程式在 `hetero_p2p/`。用 Play2Perfect 自己的環境跑（Isaac Lab 2.3.2.post1、repo 自帶有 SAPG 的 rl_games），在 Play2Perfect 的資料夾裡執行。

**官方第二階段怎麼做**：每個任務從釋出的第一階段（「玩」）策略開始，各自微調。YAML 沒寫清楚規模，但釋出的第二階段存檔寫得很清楚：四個都是 12,288 個環境、SAPG 6 組、每組 2048（存檔裡最後一次收集有 12,288 個環境，SAPG 那一欄有 6 個值、各 2048 個環境），而且輪數從 0 算，也就是只載入「玩」的權重，沒有接續它的計數器。這裡的預設就照這樣。YAML 的每組 4096 只有 3 組，「玩」的策略有 6 組，載不進去。

**聯集場景**，跟 Factory 一樣：

- 每個環境放齊四個任務的零件、孔、目標標記。
- 孔和目標標記是 kinematic。停放時放在桌面底下（z = −1 m），官方對「隨機目標」的環境也是這樣把孔藏起來。
- 零件是動態的。停放時懸在 2.5 到 4 m 高的地方，用一個等於自身重量的往上力撐住，所以不碰任何東西。每次重設都把位置寫回去，避免數值誤差讓它飄走。
- `ActiveRigidObject` 代替官方的 `env.object`／`env.hole`／`env.goal_viz`：讀的時候拿每個環境自己任務的零件，寫的時候只寫那一個。所以官方的觀測、獎勵、重設、結束判定都不用改。
- 每個環境不同的值，都從官方的問題清單讀，不手抄：零件大小（觀測）、孔的高度、子目標位置（補齊到 10 個，鎖螺絲的數量）、子目標數、零件質量。

**策略的輸入**：policy 和 critic 的輸入最後加任務 one-hot（4 個值），放在 SAPG 那一欄前面。釋出的「玩」存檔為它擴充，新欄權重是 0，所以輸出不變（檢查過：完全一樣）。

**任務分配**：

- Hetero：環境 i 整個訓練都做任務 i % 4。這個環境是一個一個重設的，不用輪換；每個 SAPG 組裡四個任務一樣多。
- Homo：全部環境做同一個任務，每 40 輪（640 步，約一個回合）在一次收集的邊界換任務；被切斷的回合分開算。

| 檔案 | 內容 |
|---|---|
| `hetero_p2p/problem_table.py` | 從官方問題清單讀四個任務的值，做成每個任務一列的張量 |
| `hetero_p2p/active_views.py` | `ActiveRigidObject`：每個環境在用的零件；停放；停放零件的抗重力 |
| `hetero_p2p/p2p_env_cfg.py`、`p2p_env.py` | 聯集場景的環境：繼承官方 `PreciseAssemblyEnv`，改的地方標 `HETERO` |
| `hetero_p2p/checkpoint_utils.py` | 為 one-hot 擴充釋出的存檔（新欄權重 0） |
| `hetero_p2p/train_utils.py` | 每輪的紀錄（每個任務、每個 SAPG 組的回合）、輕量存檔、Homo 換任務 |
| `scripts/train_p2p.py` | 訓練，改自 Play2Perfect 的 `isaacsimenvs/train.py` |
| `scripts/eval_p2p.py` | 每個任務的成功率，用官方的離線評估方法（每個環境只算第一回合、不隨機、關掉擾動） |
| `scripts/check_p2p_env.py` | 檢查：停放、在用的零件、Homo 換任務 |
| `scripts/plot_p2p.py` | 每個任務的成功率、獎勵對步數和時間（全部環境，或只看 SAPG 最後一組） |
| `hetero_p2p/pose_record_p2p.py`、`scripts/record_video_p2p.py` | 錄影，做法跟 Factory 一樣：PhysX 記位置（停放的零件不畫），再用 `render_offline.py --ground_z 0` 畫。匯出網格的程式現在也讀 USD 的方塊、圓柱、球（Play2Perfect 的桌子、L 型插銷和孔都是方塊） |

指令見英文版。

目前檢查過的（2026-10-09，CPU 物理、24 個環境；GPU 正在跑 Factory 的計時）：

- 擴充後的「玩」策略，動作、動作隨機性、value、LSTM 狀態都跟原本的完全一樣。
- 場景建得起來；輸入是 144（policy）和 166（critic）個值；one-hot 對得上每個環境的任務；子目標數 2／2／2／10。
- 停放的零件 60 步內偏離不到 0.1 µm。
- Homo 換任務時，每個環境都換到下一個任務，one-hot、子目標數、零件大小跟著換；Homo 每次收集只有一個任務。
- 訓練跑得起來，紀錄和存檔都有寫。
- CPU 上驗不了的：PhysX 的 CPU 版不會保留寫給 kinematic 物體的位置（桌子、孔會跳回原位），所以在用的零件最後放在哪，要在 GPU 上驗。

GPU 上檢查過的（2026-10-10）：

- 停放的零件不動；在用的孔在桌上（z 約 0.53 m），在用的零件落在桌上或孔上；Homo 換任務正常。
- **官方四個組裝策略（為 one-hot 擴充）放進聯集場景，用官方離線評估的方法（每個任務 1,024 個環境），成功率跟官方一樣**：

  | 任務 | 聯集場景 | 官方（`eval_offline.py`，同一台機器） |
  |---|---|---|
  | 緊配合插入 | 95.9% | 96.9% |
  | 橫樑組裝第一步 | 97.3% | 97.1% |
  | 橫樑組裝第二步 | 92.3% | 92.2% |
  | 鎖螺絲 | 61.0% | 61.4% |

  每個官方策略只會自己的任務（別的任務 0% 到 25%）。
- 兩個官方策略的影片：桌子、治具、零件位置都對；插銷插進去、桌腳鎖上去。
- 12,288 個環境的速度（一次收集 16 步的模擬時間）：Homo 緊配合插入約 1.0 秒、橫樑第一步約 2.2 秒、第二步約 1.5 秒、鎖螺絲 8.5 到 11.6 秒；Hetero 約 9 秒。**混在一起跟最慢的任務（鎖螺絲）一樣慢**，跟 Factory 小規模時一樣，但這裡約是 Homo 平均的 2.4 倍（Factory 只多 11%）。建場景約 11 分鐘；GPU 記憶體最多約 48 GB。

訓練（2026-10-10 到 10-13，結果見 [results/README.zh-TW.md](results/README.zh-TW.md)）：Hetero、Homo 各 10,000 輪（20 億步，每個任務約等於單獨訓練 2,500 輪；官方存檔每個任務練了 7,000 到 14,000 輪，2 到 5 小時學會，約 1,100 到 2,700 輪），之後每 2,000 輪用官方離線方法評估、錄影。

跟官方第二階段不一樣的地方：任務 one-hot（和擴充的存檔）；聯集場景和停放；固定的輪數（官方設定是跑到手動停）。

## 還要組裡決定的事

- 回合長度：統一 30 秒，還是各任務用官方的長度。
- Homo 換任務的方式。
- Hetero 要不要按任務標準化 advantage（官方做法對 Hetero 不利）。
- 計時要用訓練好的策略。亂數動作會低估混在一起的代價：亂數動作量到 6% 到 9%，訓練時是 12%。
- 10/07 開會提的：更多任務（例如 128 種）、off-policy、改 reward，這些都超出 RULES。

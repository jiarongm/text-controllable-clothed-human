# 文字可控之自由拓樸三維著衣人體生成與即時姿態驅動系統

> Text-Controllable Free-Topology 3D Clothed Human Generation and Real-Time Pose-Driven System
>
> 國立東華大學 資訊工程學系 碩士論文｜江佳蓉｜2026

本專案串接單目相機姿態估測（ROMP）與擴散式著衣人體生成（Chupa），
在「離線生成、即時驅動」的架構下，讓使用者輸入身高體重即可得到個人化的著衣人體網格，
並以文字提示更換服裝，最終由攝影機即時驅動。
[文字可控之自由拓樸三維著衣人體生成與即時姿態驅動系統](https://hdl.handle.net/11296/r3wt66)

---

## ⚠️ 關於第三方素材

本 repo **不包含**任何模型權重、第三方資料集、第三方原始碼、生成網格或影片素材，
因為它們的授權條款不允許再散布。

**請先閱讀 [THIRD_PARTY.md](THIRD_PARTY.md)**，依其說明自行取得
SMPL-X 模型、Chupa 原始碼與權重、RenderPeople 模板等素材後，本 repo 的程式即可執行。

本 repo 包含的是：本人撰寫的全部程式碼、本人量測的實驗數據與說明文件。

---

## 系統總覽

```
攝影機 ──► ROMP ──► SMPL (72維 thetas) ──┐
                                          ├──► LBS 線性混合蒙皮 ──► 即時著衣人體
身高/體重 ──► betas 擬合 ──► SMPL-X ──► Chupa ──► 著衣網格 ──► 蒙皮權重轉移 ──┘
                                      │
                           文字提示 ──┘        UV 貼圖生成（ControlNet + SD1.5）
```

- **離線階段**（約 10 分鐘）：由身高體重擬合體型參數 → Chupa 生成著衣網格 → 蒙皮權重轉移 → UV 貼圖生成
- **即時階段**：ROMP 逐影格估測姿態 → 座標校正 → LBS 變形 → 去頭合成顯示

「自由拓樸」指著衣網格由擴散模型雕刻而成，不受固定模板的頂點拓樸限制，
因此裙裝、外套等與人體拓樸不同的服裝也能表現；代價是網格沒有現成的蒙皮權重，
必須另行從 SMPL-X 轉移（見 `src/skinning/`）。

## 目錄結構

```
src/
  core/           蒙皮核心與 ROMP→Chupa 參數轉換
  realtime/       Gradio 即時介面與伺服端
  body_shape/     身高體重 → betas 擬合、體型資產建立、SMPL↔SMPL-X 轉換
  skinning/       蒙皮權重轉移四方法與比較實驗
  appearance/     UV 展開、多視角貼圖生成與烘焙
  experiments/    批次消融與參數掃描
  evaluation/     量測與效能評估（第四章數據來源）
  pipeline/       流程編排與 ROMP 動作前處理
  driving/        離線影片驅動與渲染
  figures/        論文圖表繪製
data/
  results/        實驗數據（CSV / JSON，純數值）
docs/
  chupa_modifications.md    Chupa 原始碼需套用的修改（文字描述）
tools/
  assemble_repo.py          白名單式 repo 整理
  audit_repo.py             推送前著作權稽核
  export_betas_csv.py       體型參數匯出
```

## 環境

本專案使用三個獨立的 conda 環境，因為 Chupa 與 ROMP 的 PyTorch 版本不相容：

| 環境 | Python | PyTorch | 用途 |
|---|---|---|---|
| `romp` | 3.10.20 | 2.5.1+cu121 | ROMP 姿態估測、即時驅動、Gradio 介面 |
| `chupa` | 3.8.20 | 1.13.1+cu117 | Chupa 著衣網格生成、蒙皮權重轉移 |
| `chupa_tex` | 3.8.20 | 1.13.1+cu117 | UV 貼圖生成（較新版 diffusers） |

```bash
# romp 環境
conda create -n romp python=3.10 && conda activate romp
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install simple-romp==1.1.4 gradio trimesh numpy

# chupa 環境：依 Chupa 官方 repo 的說明建立（本 repo 不含其程式碼）
```

## 執行流程

### 1. 建立個人化體型資產（離線）

```bash
conda activate chupa

# 由身高體重擬合 10 維 betas
python src/body_shape/fit_betas_from_measurements.py --height 165 --weight 55

# 建立 subject 參數檔（需已取得 RenderPeople A-pose 模板，見 THIRD_PARTY.md）
python src/body_shape/make_subject_from_measurements.py --height 165 --weight 55

# 以文字提示生成著衣網格（需已套用 docs/chupa_modifications.md 的修改）
python -m scripts.chupa --config configs/renderpeople.yaml \
    --text "a woman wearing a red hoodie and black long pants" \
    --subject rp_lib_165_55_001
```

### 2. 蒙皮權重轉移

```bash
python src/skinning/export_source_smplx.py --subject rp_lib_165_55_001
python src/skinning/transfer_chupa_skinning_weights.py --method m2   # k-NN 高斯，k=12
```

### 3. UV 貼圖生成

```bash
conda activate chupa_tex
python src/appearance/build_asset.py --match red_hoodie --work demo_work/my_case
```

### 4. 即時驅動

```bash
conda activate romp
python src/realtime/gradio_realtime_romp_chupa.py
```

介面開啟後請以 `http://127.0.0.1:7860` 連線
（瀏覽器僅在 localhost 或 HTTPS 下允許存取攝影機）。

## 實驗重現

| 實驗 | 指令 | 對應章節 |
|---|---|---|
| 蒙皮四方法比較 | `python src/skinning/compare_skinning_methods.py` | 4.2 |
| 表面粗糙度量測 | `python src/evaluation/measure_roughness.py` | 4.2 |
| 效能分階段拆解 | `python src/evaluation/profile_pipeline.py` | 4.3 |
| 文字控制消融 | `python src/experiments/batch_text_ablation.py` | 4.4 |
| 參數掃描 | `python src/experiments/run_factor_sweep.py` | 4.4 |
| 拉普拉斯權重掃描 | `python src/experiments/run_laplacian_sweep.py` | 4.2 |
| 體型圍度量測 | `python src/evaluation/measure_girth.py` | 4.7 |
| 九組案例生成 | `python src/experiments/batch_outfit_cases.py` | 4.8 |
| 九組案例貼圖 | `python src/experiments/batch_texture_cases.py` | 4.8 |

實驗產出的數值紀錄於 `data/results/`，可直接與論文表格對照。

## 主要發現

1. **蒙皮權重轉移**：k-NN 高斯加權（k=12）在全域指標上明確優於最近頂點、重心座標與部位約束三種方法；真正的權衡在全域平滑與局部邊界之間，而非單一方法全面最佳。
2. **文字控制**：Chupa 的 resample 階段以裸體先驗為條件，會抹除文字生成的服裝；必須將 `n_resample_from_text` 與 `n_resample` 同時設為 0，服裝類型才由文字決定，且與 CFG 強度無關。
3. **身形擷取**：ROMP 輸出的 betas 在實測影片中均小於 0.02，無法反映受測者體型；改以使用者輸入的身高體重擬合 betas 方屬可行。

## 影片素材

Demo 影片與實驗用影片因檔案大小與人物肖像權考量，不放在 repo 中。

## 授權

本 repo 中**由本人撰寫的程式碼與數據**以 MIT License 釋出（見 [LICENSE](LICENSE)）。
本授權不涵蓋任何第三方素材；第三方素材的授權依其各自條款，見 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 引用

```bibtex
@mastersthesis{chiang2026clothed,
  title  = {文字可控之自由拓樸三維著衣人體生成與即時姿態驅動系統},
  author = {江佳蓉},
  school = {國立東華大學資訊工程學系},
  year   = {2026},
  type   = {碩士論文}
}
```

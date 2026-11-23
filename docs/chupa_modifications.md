# Chupa 原始碼需套用的修改

Chupa 的 repo（<https://github.com/snuvclab/chupa>）未附任何 LICENSE 檔案，
依預設即為「保留一切權利」，因此**本 repo 不包含其原始碼，也不提供 diff**
（diff 仍會重現原始碼片段）。

以下以文字描述本論文對 Chupa 所做的修改。請自行下載原始 repo 後依此套用。

---

## 修改一：固定 resample 參數（必要）

**檔案**：`scripts/chupa.py`
**位置**：`difftext_sample` 方法的簽章，以及類別初始化中讀取 resample 設定之處

**原因**（論文 3.6.2 與 4.4 節）：
Chupa 的生成流程在兩處執行以**裸體先驗為條件**的 resample，
這兩處會把文字提示生成出來的服裝抹除，使輸出退回接近裸體的形狀。

證據鏈（論文 4.4.2）：

- `normal_F` 有裙 → `normal_F_re` 無裙（第一處，文字階段）
- `mesh_dual` 有裙 → `mesh_refine` 無裙（第二處，精修階段）

2×2 消融進一步證實：服裝類型**完全由 resample 決定，與 CFG 強度無關**
（`cfg=2.0` 搭配 `n_resample=0` 仍能生出短褲）。

**改法**：將兩處 resample 次數都設為 0。

1. `difftext_sample` 方法的預設參數中，把 `n_resample` 的預設值改為 `0`
   （原值為 `5`）。此處對應論文中的 `n_resample_from_text`。
2. 類別中控制精修階段 resample 次數的屬性（本論文中為 `self.n_resample`）
   亦設為 `0`。

本論文最終採用的固定組態：

| 參數 | 值 |
|---|---|
| `n_resample_from_text` | 0 |
| `n_resample` | 0 |
| `text_steps` | 100 |
| `text_cfg_scale` | 7.5 |
| `text_image_cfg_scale` | 1.5 |

---

## 修改二：逐階段診斷輸出（選用，用於重現論文圖表）

**檔案**：`scripts/chupa.py`
**位置**：`difftext_sample` 方法內部

**用途**：論文 4.4 節的逐階段消融圖需要 stage1 / stage2 / stage3 各自的法向量圖。
原始程式只保留最終輸出，中間結果會被覆寫。

**改法**：在 `difftext_sample` 方法開頭，由文字提示與參數組出一個安全的檔名標籤，
並在三個階段各自把法向量圖另存到 `debug_ablation/` 目錄下。

標籤的組法（避免檔名含空白或特殊字元）：

```python
safe_prompt = "_".join(str(text).split())
safe_prompt = "".join(c for c in safe_prompt if c.isalnum() or c in "_-")[:60]
debug_tag = f"{safe_prompt}_seed{seed}_cfg{cfg_scale}_nres{n_resample}"
debug_dir = Path("debug_ablation")
debug_dir.mkdir(exist_ok=True)
```

然後在 stage1（文字生成）、stage2（resample 後）、stage3（精修後）三個位置，
各加一行把當前的法向量圖以 `{debug_tag}_stage{N}.png` 存入 `debug_dir`。

> **注意**：若沿用原始碼的縮排習慣，`difftext_sample` 的 `def` 需縮排 **4 格**
> （類別方法層級）。本論文實作過程曾因縮排 3 格而產生 `IndentationError`。

---

## 修改三：拉普拉斯平滑權重（實驗用）

**檔案**：精修階段的組態檔（`configs/` 下對應的 yaml）
**改法**：調整 neural deferred shading 的拉普拉斯正則化權重。

本論文比較了權重 40 與 100 兩組設定（論文 4.2 節表 4-2）。
權重越大，表面越平滑但服裝細節越少。

---

## 未修改的部分

以下屬於 Chupa 原有行為，本論文未改動，但在論文中有討論：

- `scripts/chupa.py` 中的 `self.smpl_gender` 為**死參數**
  （以 grep 確認整個 repo 無任何處讀取它），因此無法透過它支援性別差異。
  論文 4.7 節另有說明：SMPL-X neutral 模型的 10 個 beta 只編碼高矮胖瘦，
  不含性別維度。
- 兩階段架構（法向量圖擴散 → 多視角精修 → 雕刻）本身未改動。

# 第三方素材與取得方式

本專案建立在數個第三方模型、資料集與程式碼之上。它們的授權條款**不允許再散布**，
因此本 repo 一律不包含這些檔案，只說明如何自行取得與放置位置。

依下列步驟取得後，本 repo 的程式即可重現論文中的結果。

---

## 1. SMPL-X 人體模型

| 項目 | 內容 |
|---|---|
| 來源 | <https://smpl-x.is.tue.mpg.de/> |
| 取得方式 | 註冊帳號 → 同意授權 → 下載 `SMPLX_NEUTRAL.npz` |
| 授權 | 僅限非商業研究用途，**明文禁止複製、分享、散布、轉售、轉授權（全部或部分）** |
| 本 repo 是否包含 | **否** |

放置位置：

```
src/smpl_related/models/smplx/SMPLX_NEUTRAL.npz
```

本專案另需 SMPL（24 關節）模型用於 ROMP 輸出的解讀，
來源 <https://smpl.is.tue.mpg.de/>，授權條款相同。

### SMPL → SMPL-X 參數轉換

本論文 3.3 節使用官方 `smplx` repo 的 `transfer_model` 工具，
取得自 <https://github.com/vchoutas/smplx>（授權見該 repo）。
執行時需要的 `smpl2smplx_deftrafo_setup.pkl` 等對應檔隨 SMPL-X 授權一併下載。

本 repo 的 `docs/setup_transfer_model.md` 記錄了實際執行時遇到的三個相依性問題
（`chumpy` 相依、namedtuple 存取、`var_dict.update`）與修正方式，
但不包含該工具的原始碼。

---

## 2. Chupa（著衣人體生成）

| 項目 | 內容 |
|---|---|
| 來源 | <https://github.com/snuvclab/chupa> |
| 論文 | Kim et al., *Chupa: Carving 3D Clothed Humans from Skinned Shape Priors using 2D Diffusion Probabilistic Models*, ICCV 2023 |
| 授權 | **該 repo 未附任何 LICENSE 檔案**，依預設即為「保留一切權利」 |
| 本 repo 是否包含 | **否**，連修改過的版本也不包含 |

本論文對 Chupa 的 `scripts/chupa.py` 做了修改（固定 resample 參數、加入逐階段診斷輸出）。
由於原始碼不可散布，**本 repo 不提供修改後的檔案，也不提供 diff**
（diff 仍會重現原始碼片段），而是在 `docs/chupa_modifications.md`
以文字描述需要改動的位置與改法，供自行套用。

Chupa 的預訓練權重與其訓練所用的 RenderPeople 資料集皆不在本 repo 中。
原作者亦說明因授權問題無法分享 RenderPeople 資料集。

---

## 3. RenderPeople 資料集

| 項目 | 內容 |
|---|---|
| 來源 | <https://renderpeople.com/> |
| 授權 | 商業授權，依購買條款，**不可再散布** |
| 本 repo 是否包含 | **否** |

本論文使用 Chupa 隨附的 RenderPeople subject 參數檔（`data/renderpeople/smplx/*.pkl`）
作為 A-pose 模板。這些 `.pkl` 含 RenderPeople 的姿勢、尺度與位移資料，故不包含在本 repo。

**可以發布的部分**：本論文自行擬合出的體型參數（10 維 SMPL betas）為本人計算結果，
整理於 `data/results/shape_library_betas.csv`，不含任何 RenderPeople 原始資料。
取得上述 `.pkl` 模板後，執行 `src/body_shape/make_subject_from_measurements.py`
即可用該 CSV 的 betas 重建本論文使用的 subject 檔。

---

## 4. ROMP（即時姿態估測）

| 項目 | 內容 |
|---|---|
| 來源 | <https://github.com/Arthur151/ROMP>（PyPI: `simple-romp` 1.1.4） |
| 授權 | 非商業研究授權，程式碼可透過 pip 安裝 |
| 本 repo 是否包含 | **否**（以 `pip install simple-romp==1.1.4` 安裝） |

ROMP 的預訓練權重於首次執行時自動下載，不包含在本 repo。

---

## 5. 擴散模型與貼圖生成相依套件

| 項目 | 來源 | 本 repo 是否包含 |
|---|---|---|
| Stable Diffusion v1.5 | <https://huggingface.co/runwayml/stable-diffusion-v1-5>（CreativeML Open RAIL-M） | 否 |
| ControlNet | <https://github.com/lllyasviel/ControlNet>（Apache-2.0，權重另有條款） | 否 |
| nvdiffrast | <https://github.com/NVlabs/nvdiffrast>（NVIDIA Source Code License，非商業） | 否 |
| xatlas | <https://github.com/jpcy/xatlas>（MIT） | 否，以 `pip install xatlas` 安裝 |

權重於首次執行時由 Hugging Face 自動下載。

---

## 6. 本論文使用的參考文獻

參考文獻的 PDF 受出版社著作權保護，**本 repo 不包含任何論文 PDF**。
`docs/references.md` 提供完整引用資訊與 DOI / arXiv 連結，可自行取得。

---

## 7. 輔助軟體

本專案開發過程使用的第三方軟體（Blender、MeshLab、FFmpeg 等）皆為各自獨立散布，
**本 repo 不包含安裝檔**。`docs/environment.md` 列出使用到的版本與官方下載連結。

---

## 本 repo 實際包含什麼

- 本人撰寫的所有 Python 程式碼（蒙皮權重轉移、即時驅動、體型擬合、UV 貼圖烘焙、批次實驗、圖表繪製）
- 本人量測的實驗數據（CSV / JSON 數值）
- 論文原始檔（Markdown）
- 說明文件與環境設定紀錄

## 本 repo 不包含什麼

模型權重、第三方資料集、第三方原始碼、生成的三維網格與貼圖、
影片素材、論文 PDF、第三方軟體安裝檔。

生成的網格與貼圖雖由本專案程式產生，但其內容衍生自 Chupa 的預訓練權重
（該權重以 RenderPeople 資料集訓練），衍生物的授權狀態不明確，
因此一併排除，只在論文中以圖片形式呈現結果。

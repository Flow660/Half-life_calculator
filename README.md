# 🧬 SILK Protein Half-life Calculator / タンパク半減期計算アプリ

**SILK（Stable-Isotope-Labelling Kinetics）の pulse / pulse&chase データから、タンパクごとの半減期を推定する Python + Streamlit アプリです。** 前駆体の標識動態を考慮したモデルと単一指数モデルを選択でき、推定結果の可視化、品質評価（QC）、半減期回復シミュレーション、結果のエクスポートに対応します。

A Python + Streamlit application for estimating protein turnover half-lives from SILK pulse / pulse-and-chase data, with precursor modelling, fit-quality assessment and noise-driven recovery simulations.

> **この文書の範囲：** 添付の `app.py` と既存READMEに基づく操作・機能説明です。入力ローダーや計算コアの内部仕様は既存READMEの記載を引き継いでおり、各モジュールの実ファイルやサンプルデータによる数値検証は行っていません。

---

## 1. Quick start / クイックスタート

プロジェクトのルートディレクトリで実行します。

```bash
pip install -r requirements.txt
streamlit run app.py
```

起動後、ブラウザーに表示されるアプリへアクセスします。通常のローカル接続先は `http://localhost:8501` です。実際の接続先は起動時の表示を確認してください。

1. サイドバーで **Use bundled sample data / サンプルデータを使用** をオンにするか、解析に必要なファイルをアップロードします。
2. **Kinetic model** を選択します。
3. 2-compartment モデルの場合は **Precursor source** を選択します。
4. **Fit mode** と必要なオプションを設定します。
5. **▶ Run / 計算実行** を押します。

既存READMEに記載されている起動用スクリプトがプロジェクトに含まれている場合は、macOS / Linux では `./run.sh`、Windows では `run.bat` も使用できます。

---

## 2. Inputs / 入力ファイル

### 2.1 ファイルの種類

| ファイル | 役割 |
|---|---|
| **Hsol Pulse** | Pulse期間の可溶性前駆体（遊離重標識アミノ酸）の Label/Total 比の時系列。タンパク標識を駆動する前駆体曲線に使用します。 |
| **Hsol Pulse&Chase** | Pulse&Chase条件における前駆体比の時系列。 |
| **Hpoi Pulse** | Pulse条件における各タンパクの Label/Total 比の時系列。 |
| **Hpoi Pulse&Chase** | Pulse&Chase条件における各タンパクの Label/Total 比の時系列。 |

アップロード欄は `.xlsx` と `.xls` を選択可能です。`.xls` の実際の読み込み可否は、入力ローダーとインストール済みのExcel読み込みエンジンに依存します。サンプルファイルは `.xlsx` 形式です。

### 2.2 設定別の必須ファイル

**4ファイルすべてが常に必須というわけではありません。** `app.py` の実行前チェックは次のとおりです。

| モデル | 前駆体の求め方 | Fit mode | 実行前チェックで必須のファイル |
|---|---|---|---|
| Single-exponential | 使用しない | Pulse only | Hpoi Pulse |
| Single-exponential | 使用しない | Pulse + Pulse&Chase | Hpoi Pulse + Hpoi Pulse&Chase |
| 2-compartment | Plasma measurement | Pulse only | Hsol Pulse + Hpoi Pulse |
| 2-compartment | Plasma measurement | Pulse + Pulse&Chase | Hsol Pulse + Hpoi Pulse + Hpoi Pulse&Chase |
| 2-compartment | Global fit | Pulse only | Hpoi Pulse |
| 2-compartment | Global fit | Pulse + Pulse&Chase | Hpoi Pulse + Hpoi Pulse&Chase |

- **Hsol Pulse&Chase はUIの実行前チェックでは必須ではありません。** 前駆体のchase測定値がある場合は追加してください。省略時の詳細な計算挙動は計算コア側に依存します。
- Global fitでは、Hsolデータは前駆体推定ではなく、測定値との比較表示に利用される設計です。
- サンプル利用時は、`sample_data/` に4つのサンプルファイルが配置されている必要があります。

### 2.3 Excelの想定レイアウト

以下は既存READMEに記載された入力仕様です。

| 種類 | A列 | B列 | C列 | D列以降 |
|---|---|---|---|---|
| Hpoi | ProteinGroups | ProteinDescriptions | Gene Name | 時点ラベル（日） |
| Hsol | 先頭3列は前駆体時系列の測定値として使用しない想定 | 同左 | 同左 | 時点ラベル（日） |

- 1行目をヘッダーとします。
- Hsolでは2行目に各時点の前駆体比を配置します。
- Hpoiでは2行目以降をタンパクごとの測定値とします。
- 比は **Label / Total の分率** として扱います。表示上の百分率切り替えと、入力値の単位を混同しないでください。
- 既存仕様では、`t = 0, ratio = 0` の初期点が自動追加されます。明示的な0日データとの重複処理はローダー側の確認が必要です。

---

## 3. Models and fit settings / モデル・推定設定

### 3.1 2-compartment（precursor-driven）— 初期選択

前駆体の標識が徐々に上昇する動態を考慮するモデルです。既存READMEでの前駆体pulse曲線の表記は次のとおりです。

$$
H_{sol}(t)=1-Ae^{-t/T_1}-(1-A)e^{-t/T_2}
$$

タンパクごとの推定パラメーターを時間定数 $\tau$ とすると、半減期は

$$
t_{1/2}=\ln(2)\,\tau
$$

です。前駆体動態を無視できないデータに対して、その影響を取り込むための選択肢です。モデルを選んだだけで推定値の正確性が保証されるわけではなく、前駆体推定、時点数、欠測、適合品質を併せて評価してください。

### 3.2 Single-exponential pool

前駆体が瞬時に標識されると仮定します。Pulse期間の標識比は

$$
LR(t)=1-e^{-kt}
$$

chase切り替え時刻を $t_p$ としたchase期間の標識比は

$$
LR(t)=LR(t_p)e^{-k(t-t_p)}
$$

で表され、半減期は

$$
t_{1/2}=\frac{\ln(2)}{k}
$$

です。前駆体の上昇が遅いデータでは、前駆体動態とタンパク代謝を切り分けられない可能性があります。モデル間の結果の差を評価し、特定タンパクの半減期が必ず一定方向に変化するとは仮定しないでください。

### 3.3 Precursor source / 前駆体の求め方

2-compartmentモデルで選択します。

| 選択肢 | 説明 |
|---|---|
| **Plasma measurement (Hsol files)** — 初期選択 | 実測Hsol曲線から前駆体パラメーターを推定します。Hsol Pulseが必要です。 |
| **Global fit of all proteins (paper method)** | タンパク標識時系列から共通の前駆体パラメーターを推定します。Hsolを与えた場合は測定値との比較表示に使用します。 |

**識別性への注意：** 少数のpulse時点だけでは、「前駆体の標識が遅い」場合と「タンパクの代謝が遅い」場合を区別しにくくなります。Global fitの最適値が探索範囲の境界に達した場合、アプリは警告を表示します。適切なHsol実測値がある場合は、その測定値との整合性も確認してください。

画面の “paper method”、Fig. 3i、Supplementary Fig. 4b などは、既存READMEおよびアプリ中の Fornasiero et al. (2018) への参照表記に由来します。本更新では原論文との数値的一致を検証していません。

### 3.4 Fit mode / フィット対象

- **Pulse only** — 各タンパクのpulse時系列をフィットします。
- **Pulse + Pulse&Chase** — PulseとPulse&Chaseを同時にフィットします。

Chase切り替え時刻 **tpoint** は自動検出を選択するか、日単位で手動設定します。手動入力の初期値は **14日** です。自動検出の詳細な判定ロジックは計算コア側にあります。

### 3.5 Quick preview

**Quick preview (first 100 proteins only)** をオンにすると、最大100タンパクで計算します。全タンパクの解析結果とは区別してください。Global fitでは前駆体の推定に用いるタンパク集合も変わる可能性があります。

---

## 4. Results / 結果表示

計算後、タンパク数、有限な半減期の中央値、chase切り替え時刻、モデル、R²中央値が表示されます。

| タブ | 主な機能 |
|---|---|
| **Overview** | 前駆体曲線、推定パラメーター、Hsolとの適合・比較、半減期分布。ヒストグラムの横軸は対数表示を切り替え可能。 |
| **Lifetime curve (Fig 3i)** | 半減期と標識比の関係を、モデル曲線とタンパク測定点で表示。軸範囲、百分率／分率、時点、pulse／chase、測定点表示を調整可能。 |
| **Results table** | 遺伝子・タンパクグループ・説明文による検索、並べ替え、Excel／CSV出力。 |
| **Protein detail** | タンパクを選択し、測定点とフィット曲線、半減期、SE、95% CI、R²、品質注記を表示。 |
| **Fit quality (QC)** | 品質指標、閾値による採否判定、欠測率、QC表出力、残差ブートストラップ。 |
| **100-day simulation** | 実データ由来のノイズ・欠測条件で既知の半減期を再推定し、回復性能を評価。 |

### 主な結果列

| 列名 | 意味 |
|---|---|
| `HalfLife_days` | 推定半減期（日） |
| `HalfLife_SE_days` | 半減期の標準誤差（日） |
| `CI95_low_days`, `CI95_high_days` | 95%信頼区間の下限・上限（日） |
| `FitParameter` | モデルに対応する推定パラメーター |
| `R2` | 決定係数 |
| `SSE` | 残差平方和 |
| `Note` | 境界への張り付きやSE推定不能などの品質注記 |

既存READMEによると、標準誤差・通常の95% CIは、目的関数の局所曲率に基づく漸近近似とデルタ法で算出します。高不確実性の推定ではCI下限が負になる場合があり、その値は自動的に0へ切り詰められません。

---

## 5. Fit quality / 品質評価（QC）

QCタブでは各タンパクの品質を評価し、基準への適合を `QC_pass` と判定理由 `QC_reason` で示します。

### 5.1 初期設定

| 項目 | 初期値・条件 |
|---|---|
| Min R² | 0.90 |
| Max relative 95% CI | 0.50 |
| Relative-CI basis | **± half-width** |
| Min observed points | 3 |
| Max missing fraction | 0.50 |
| Exclude estimates pinned to a fit bound | オン |
| 有限な95% CIを要求 | 有効 |
| Half-life ceiling | 0（上限なし） |

**「相対95% CIが50%」の意味は、CIの全幅ではなく半幅が初期選択です。**

$$
\mathrm{RelCI}_{halfwidth}=\frac{CI_{high}-CI_{low}}{2\,t_{1/2}}
$$

全幅を選んだ場合は

$$
\mathrm{RelCI}_{width}=\frac{CI_{high}-CI_{low}}{t_{1/2}}
$$

となります。例えば半減期20日、CIが12–28日の場合、半幅比は0.40、全幅比は0.80です。同じ閾値0.50でも判定が異なります。

### 5.2 品質表とエクスポート

品質表には、R²、RMSE、CI、相対CI、CV、観測点数、欠測率、境界フラグなどが表示されます。`N_peptides` / `N_replicates` 列も存在する場合に表示されますが、**添付のUIにはメタデータ専用のアップロード欄はありません。**

- **Show retained proteins only** は初期状態でオンです。
- QCのExcel／CSV出力には、現在の表示対象 `view_q` が渡されます。全件を出力したい場合はこのチェックを外してください。
- QCフィルターはQC表の判定・表示に適用されます。元の **Results table** 自体を置き換える処理ではありません。

### 5.3 Residual bootstrap / 残差ブートストラップ

QC通過タンパクについて残差を再標本化し、percentile CIを追加計算できます。

| 設定 | 初期値 |
|---|---|
| Bootstrap replicates | 400 |
| Max proteins (retained subset) | 500 |
| Seed | 0 |

ブートストラップ結果は通常の漸近CIとの比較用として表示されます。通常のQC判定にブートストラップCIが自動的に置き換わる処理は、添付UIにはありません。

**方法欄に記載する際は**、R²と相対CIだけでなく、半幅／全幅の定義、観測点数、欠測率、境界除外など、実際に適用した条件も記録してください。画面の提案文は簡略版です。

---

## 6. Recovery simulation / 半減期回復シミュレーション

**“100-day simulation” はタブ名であり、測定期間を100日に延長する機能ではありません。** 実データの測定時点で既知の半減期を持つ模擬データを作成し、同じモデルで再推定します。100日を超える真の半減期も指定できます。

### 6.1 基本手順

1. 現在の解析結果のモデルと、該当する場合は推定済み前駆体曲線を使います。
2. 指定した真の半減期ごとに、実測時点での標識曲線を生成します。
3. 実データ由来のノイズと欠測条件を反映します。
4. 模擬データから半減期を再推定します。
5. Bias、95% CI被覆率、相対誤差などから回復性能を評価します。

### 6.2 初期設定

| 設定 | 初期値 |
|---|---|
| True half-lives（日） | 0.5, 1, 5, 20, 50, 75, 100, 125, 150, 200 |
| Replicates per half-life | 2,000 |
| Random seed | 123 |
| Noise scale | 実測推定ノイズの1.0倍 |
| Extra missing per point | 0.0 |
| Robust noise (MAD) | オン |
| ノイズ推定対象 | 初期選択は解析範囲内のタンパク（R²フィルターなし） |
| Analysis range（日） | 1–100 |

ノイズ推定対象の選択肢は `simulation.NOISE_SELECTIONS` から読み込まれます。解析範囲の上下限は、解析範囲による選択が有効なときに編集できます。

### 6.3 回復範囲の判定基準

| 指標 | 初期基準 |
|---|---|
| 絶対bias（%） | 15%以下 |
| 95% CI被覆率 | 0.90以上 |
| 相対誤差の中央値 | 0.50以下 |

既存READMEでは、合格する真の半減期の最長の連続区間から下限・上限を提案する仕様とされています。詳細な区間選択ロジックは `simulation.recommend_cutoff` に依存します。

**信頼できる半減期範囲はデータと設定に依存します。** 「1〜約150日」などの固定範囲が常に保証されるわけではありません。モデル、前駆体、ノイズ推定対象、測定時点、欠測、反復数、乱数seed、判定基準と併せて、実際の実行結果を記録してください。また、同じモデルで生成・再推定するシミュレーションは、モデル自体の妥当性を独立に証明するものではありません。

### 6.4 出力

- 真の半減期ごとの回復指標を **Excel / CSV** で出力できます。
- シミュレーション図を **PNG（300 dpi）/ PDF / SVG** で出力できます。
- **Publication style — hide the figure header** をオンにすると、図の見出しを非表示にできます。設定条件は別途図注に記載してください。
- 提案された上限はQC散布図の参照表示に渡されますが、QCの **Half-life ceiling** に自動設定されるわけではありません。

---

## 7. Project layout / プロジェクト構成

以下は既存READMEに記載された構成です。今回添付されたのは `app.py` と本READMEのみです。

```text
SILK_halflife_app/
├── app.py                  # Streamlit UI
├── README.md
├── requirements.txt
├── run.sh / run.bat        # 起動用スクリプト
├── validate.py             # コア解析の検証
├── validate_extensions.py  # QC・シミュレーションの検証
├── sample_data/            # サンプルExcel 4ファイル
└── silk_core/
    ├── loaders.py          # 入力ファイル読み込み
    ├── models.py           # 速度論モデル
    ├── precursor.py        # 前駆体推定
    ├── fitting.py          # タンパクごとの推定・SE・CI
    ├── pipeline.py         # 解析処理の統合
    ├── quality.py          # QC・GOF・ブートストラップ
    ├── simulation.py       # 回復シミュレーション・範囲判定
    ├── plotting.py         # 図の作成
    └── export.py           # Excel・CSV出力
```

検証用スクリプトとサンプルが含まれている環境では、既存READMEに記載の次のコマンドで検証できます。

```bash
python validate.py sample_data
python validate_extensions.py sample_data
```

実際の依存関係は `requirements.txt` と各モジュールを確認してください。添付のUIは `numpy`、`streamlit`、`matplotlib` および `silk_core` を直接インポートします。

---

## 8. Notes and troubleshooting / 注意事項・トラブルシューティング

| 状況 | 確認事項・対応 |
|---|---|
| 必須ファイルのエラー | モデル、前駆体の求め方、Fit modeに対応するファイルがあるか確認します。 |
| サンプルデータが読み込めない | `sample_data/` と、`app.py` の `SAMPLE_FILES` に指定された4ファイルの存在を確認します。現在のUIは全ファイルの読み込み成功を個別に確認せず成功メッセージを表示するため、メッセージだけで判断しないでください。 |
| Global fitが境界に達する | 前駆体の識別性と測定時点を確認し、Hsol実測がある場合はPlasma measurementでの結果も確認します。 |
| 半減期が境界に張り付く／CIが非常に広い | 点推定だけを採用せず、`Note`、QC、観測点数、欠測率を確認します。 |
| 設定や入力を変えても結果が変わらない | 主解析は **Run** を押したときに実行されます。入力・モデルを変更後は再実行してください。 |
| 主解析をやり直した後に古い追加結果が残る | ブートストラップとシミュレーションはセッションに保存され、主解析の再実行で自動クリアされません。新しい解析結果で各処理を再実行してください。 |
| シミュレーションの基準・ノイズ設定を変更した | 保存済み結果は自動再計算されません。**Run simulation** を再度押してください。 |
| 起動時にモジュールが見つからない | `app.py` 単独では動作しません。`silk_core/` を含むプロジェクト一式と依存パッケージが必要です。 |

- 処理時間はタンパク数、モデル、環境、シミュレーション反復数に依存します。固定の実行時間は保証しません。
- 時間定数 $\tau$ と半減期 $t_{1/2}$ は異なります。本アプリの結果列 `HalfLife_days` は半減期です。
- QC通過や良好な回復シミュレーション結果は、実験・モデルのあらゆる不確実性を除去するものではありません。

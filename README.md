
# 🐱 WiLL Hamiltonian LLM A/B Test

## ✨ Overview  
This script runs an A/B test of the WiLL–Hamiltonian framework on a 100-sentence Japanese “cat” corpus. It compares:  
- **Baseline** symplectic flow  
- **WiLL** intent-dependent anchor potential  

Metrics collected:  
- 📈 Semantic Drift  
- 🏃 Momentum Norm (pnorm)  
- 🔗 Alignment Integral (A_q)  

> **⚠️ Note:** You must supply your own OpenAI API key via environment variable. This repository does **not** include any API credentials.

---

## 🚀 Quickstart

1. **Install dependencies**     
   pip install torch sentence-transformers openai numpy pandas matplotlib tqdm

2. **Set your OpenAI key**
   export OPENAI_API_KEY="YOUR_API_KEY_HERE"
  
3. **Run the test**
　　python cat_anchor_abtest_drift_pnorm_Aq_fixed_v2.py
   
4. **Inspect results in** `GPT_output2/`

   * `drift_timeseries2_GPT.png`
   * `pnorm_timeseries2_GPT.png`
   * `A_q_timeseries2_GPT.png`
   * `GPT_mean_metrics2.png`
   * `GPT_metrics2.csv`

---

## 📋 Usage Details

* **Embedding:** MiniLM-L6-v2 via `sentence-transformers`
* **Generation:** OpenAI Chat API (`gpt-4o-mini`)
* **Integrator:** Leapfrog (Störmer–Verlet) with damping & anchor
* **Output:**

  * Time-series plots
  * Summary statistics (printed)
  * Bar chart of mean metrics
  * CSV of all recorded values

---

# 🐾 README（日本語版）

## ✨ 概要

100文の日本語「猫」コーパスを対象に、WiLLハミルトニアン枠組みのA/Bテストを行います。

* **Baseline**：純粋シンプレクティック流
* **WiLL**：意図依存アンカー項追加

取得指標：

* 📈 ドリフト
* 🏃 モーメンタムノルム（pnorm）
* 🔗 整合性積分（A\_q）

> **⚠️ 注意:** OpenAI APIキーは各自で用意し、環境変数に設定してください。APIキーは含まれていません。

---

## 🚀 クイックスタート

1. **依存ライブラリをインストール**

   ```bash
   pip install torch sentence-transformers openai numpy pandas matplotlib tqdm
   ```

2. **APIキーを設定**
   export OPENAI_API_KEY="YOUR_API_KEY_HERE"

3. **スクリプトを実行**
   python cat_anchor_abtest_drift_pnorm_Aq_fixed_v2.py

4. **出力結果を確認** `GPT_output2/`

   * `drift_timeseries2_GPT.png`
   * `pnorm_timeseries2_GPT.png`
   * `A_q_timeseries2_GPT.png`
   * `GPT_mean_metrics2.png`
   * `GPT_metrics2.csv`

---

## 📋 詳細

* **埋め込み:** MiniLM-L6-v2 (`sentence-transformers`)
* **生成:** OpenAI Chat API (`gpt-4o-mini`)
* **積分子:** リープフロッグ（Störmer–Verlet）＋ダンピング＋アンカー
* **出力:**

  * 時系列プロット
  * 統計要約（コンソール出力）
  * 平均指標の棒グラフ
  * 全レコードCSV



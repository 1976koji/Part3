#!/usr/bin/env python

# ---------- 猫コーパス ----------
# =========================================
# 🐱 日本語猫コーパス（100 文）
# =========================================
cats = [
    "私は猫が好きだ。",
    "黒猫のつややかな毛並みを見ると心が落ち着く。",
    "猫のゴロゴロという喉鳴らしは最高の癒やしだ。",
    "朝日を浴びて伸びをする猫の姿が愛おしい。",
    "子猫が段ボール箱の中で丸まって眠っている。",
    "三毛猫の模様は世界に二つと同じものがない。",
    "猫は好奇心旺盛で、すぐ高い所に登りたがる。",
    "狭い隙間に器用に入り込むのは猫ならではだ。",
    "肉球のぷにぷにした感触がたまらない。",
    "雨の日、窓辺で外を眺める猫の横顔が詩的だ。",
    "夜更けに静かに歩く猫の足音はほとんど聞こえない。",
    "猫は紙袋を見ると必ずと言っていいほど潜り込む。",
    "緑色の瞳がランプのように暗闇で光る。",
    "キャットタワーのてっぺんは彼らの特等席だ。",
    "猫じゃらしを振ると目を輝かせて飛びつく。",
    "毛づくろいに余念がなく、常に身だしなみは完璧。",
    "魚の匂いがすると台所へ一直線に走って来る。",
    "ダンボールを噛んでボロボロにするのが日課。",
    "日向ぼっこをしている姿はまるで彫刻のよう。",
    "夕暮れ時、路地裏で猫同士が挨拶を交わしていた。",
    "長毛種はブラッシングすると雲のような毛が舞う。",
    "猫カフェでは時間が経つのを忘れてしまう。",
    "ご飯の時間になると鈴のような声で鳴いて催促する。",
    "ショッピングサイトで猫用おやつを大量購入した。",
    "猫は新しい段ボール箱を見つけると即座に支配する。",
    "スコティッシュフォールドの折れた耳がキュートだ。",
    "カーテンによじ登られて穴が開いた。",
    "ソファの隅で香箱座りをしている。",
    "猫は人間の言葉の抑揚をよく聞き分けるという。",
    "猫草を食べて胃を整える賢さに感心する。",
    "早朝、枕元で小さくニャーと鳴き起こしてくれる。",
    "お腹を見せてゴロンと転がるのは信頼の証。",
    "ごはん皿の音だけで飛んでくる聴覚の鋭さ。",
    "猫用のトンネルおもちゃで延々と遊んでいる。",
    "キャットウォークを作ったら部屋が立体的に使われ始めた。",
    "猫の鳴き声には18種類以上のパターンがあるらしい。",
    "フリスキーを開封すると袋の音に敏感に反応。",
    "シュレッダーした紙片の上でふみふみを始めた。",
    "キジトラ柄は森のカモフラージュ模様だ。",
    "猫の尻尾は感情のバロメーター。",
    "玄関で出迎えるときの小走りが可愛い。",
    "こたつから顔だけ出して寝落ちしている。",
    "読書を始めると必ず本の上に座る。",
    "鳥のさえずりに反応してテレビをタッチする。",
    "毛玉を吐いた後は少し申し訳なさそうな表情。",
    "砂のトイレを丁寧に掘ってから用を足す。",
    "窓ガラスに映る自分の姿にパンチを繰り出す。",
    "おもちゃのレーザーポインタを全力で追いかける。",
    "猫背と言うが、丸まった背中は優雅でもある。",
    "ねずみのおもちゃをくわえて戦利品として持ってくる。",
    "白猫は月光を浴びると幽玄な雰囲気を醸す。",
    "キッチンカウンターに乗らないでと何度注意したことか。",
    "引き出しを開ける音でおやつだと悟る。",
    "玄関マットの上で腹ばいになって人間を待つ。",
    "すり寄ってくるときの頬の柔らかさが好きだ。",
    "猫が膝に乗ると身動きがとれなくなる「猫拘束」。",
    "高速でしっぽを振るのは不機嫌のサイン。",
    "首輪の鈴がチリンと鳴るたびに所在がわかる。",
    "夜中に大運動会を開催して家が揺れる。",
    "キャットフードのパッケージを開けると目が真剣になる。",
    "布団に潜り込んで足を温めてくれる。",
    "聞き慣れない来客の声に警戒して隠れる。",
    "歯磨きガムを噛む姿が意外とワイルド。",
    "獣医さんの診察台では固まって石像のよう。",
    "トートバッグを置くと中に入り旅支度ごっこ。",
    "寝言で小さくニャッと鳴く夜もある。",
    "朝のストレッチで背中を反らす姿がヨガの達人。",
    "ふわふわの尻尾で顔を撫でられてくすぐったい。",
    "猫がいるだけで部屋の空気が柔らかくなる。",
    "段ボール迷路を作ったら探検隊長になった。",
    "キャットニップ入りのおもちゃで酔っぱらう様子が面白い。",
    "背中をトントンするとトロンと目を細める。",
    "窓辺で蝶を目で追うハンターの表情。",
    "階段を一段飛ばしで駆け上がる跳躍力。",
    "スマホの画面を猫用ゲームにすると真剣そのもの。",
    "コーヒー豆の匂いには興味を示さない。",
    "クリスマスツリーをよじ登り飾りを落とす事件。",
    "雷が鳴るとクローゼットの奥へ避難。",
    "シャンプー後のドライヤータイムは大騒ぎ。",
    "窓の結露を舐めて怒られる。",
    "長いひげは夜間のナビゲーション装置。",
    "ごろんと横になりお腹を撫でてアピール。",
    "猫規格の幅の細い棚を歩くバランス感覚。",
    "新しい匂いの服をクンクンとチェック。",
    "タブレットのペン先を狙って狩りをする。",
    "洗濯ネットを見ると病院を連想して逃げ腰。",
    "友達の犬とは適度な距離感を保つ。",
    "鍵のジャラジャラ音に反応して玄関へ。",
    "猫用ブラシを見ると喜んで頭を押し付ける。",
    "猫は家につくと言うが、人にもちゃんと懐く。",
    "箸置きを転がしてサッカーを始める。",
    "電気ストーブの前を独占して動かない。",
    "鳴き声で時刻を知らせる正確な腹時計。",
    "サングラスをかけた写真が SNS でバズった。",
    "ツンデレな態度も魅力の一部だ。",
    "録画した鳥動画をテレビに映すと大興奮。",
    "キャリーケースを見ると旅行か病院かで表情が変わる。",
    "人間のくしゃみに驚いて尻尾が膨らむ。",
    "おやつの袋を隠す場所をすでに学習している。",
    "今日も「私は猫が好きだ」と再確認した。"
]

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys, random, pathlib
import torch # Import torch here
from typing import List, Callable, Tuple # Import Callable and Tuple here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import openai
# APIキーを環境変数から取得
openai.api_key = OPENAI_API_KEY 
if not openai.api_key:
    sys.exit("環境変数 OPENAI_API_KEY を設定してください。")

# Colab ダウンロード用チェック
try:
    from google.colab import files
except ImportError:
    files = None

# ---------------- 一般設定 ----------------
EPS             = 0.005    # ← 小刻みに
DAMPING         = 0.97     # ← 摩擦強化
REFRESH_EVERY   = 200      # ← ノイズ注入頻度ダウン
SIGMA_REFRESH   = 0.002    # ← ノイズ振幅ダウン
LAMBDA_AMP      = 1.0      # ← 意思弾性項を弱める
ALPHA           = 30.0     # ← アンカー強化
EPS_DELTA       = 0.2      # ← アンカー幅拡大
SMOOTH_WIN      = 10       # ← 平滑化窓を広げる

BASE_SEED       = 1234
TRIALS          = 1

OUTDIR = pathlib.Path("GPT_output2")
OUTDIR.mkdir(exist_ok=True)

# ---------------- 埋め込みモデル ----------------
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def embed(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True).astype(np.float32)

# ---------------- テキスト生成 ----------------
def generate_text(seed: str, use_will: bool) -> str:
    prompt = (
        f"{seed}\n"
        "この文章の続きを自然な日本語で100文字程度で完結してください。"
    )
    # v1.0.0 以降の Chat API 呼び出し
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],

        temperature=0.7 if use_will else 0.7,
        top_p=0.9 if use_will else 0.9,

        max_tokens=80,
        n=1
    )
    text = resp.choices[0].message.content.strip().replace("\n", " ")
    return text

# ---------------- δ-approx ----------------
def delta_eps(q: np.ndarray, q0: np.ndarray, eps: float = EPS_DELTA) -> float:
    d = np.linalg.norm(q - q0)
    return np.exp(-d * d / (2 * eps * eps)) / (eps * np.sqrt(2 * np.pi))

# ---------------- ステップ関数 ----------------
def leapfrog_base(q, p, q0, lam):
    grad = lam * (q - q0)
    p = DAMPING * (p - 0.5 * EPS * grad)
    q = q + EPS * p
    p = DAMPING * (p - 0.5 * EPS * grad)
    return q, p


def leapfrog_will(q, p, q0, q_will, lam):
    gradV = q / (np.linalg.norm(q) + 1e-8)
    gradδ = delta_eps(q, q_will) * (-(q - q_will) / (EPS_DELTA ** 2))
    gradP = gradV
    dW = ALPHA * gradδ + gradV - lam * gradP
    p = DAMPING * (p - 0.5 * EPS * dW)
    q = q + EPS * p
    p = DAMPING * (p - 0.5 * EPS * dW)
    return q, p

# ---------------- コーパス読み込み ----------------
def load_corpus() -> list:
    try:
        cats
        return list(cats)
    except NameError:
        sys.exit("cats = [...] を定義してください。")

# ---------------- 実験関数 ----------------
def run_experiment(lambda_fn: Callable[[int], float], use_will: bool, offset: int = 0):
    random.seed(BASE_SEED + offset)
    np.random.seed(BASE_SEED + offset)
    corpus = load_corpus()
    q0 = embed(corpus[0])
    q_will = q0.copy()
    q = q0.copy()
    p = np.zeros_like(q0)

    records = []
    for step, seed in enumerate(tqdm(corpus, desc=f"{'WiLL' if use_will else 'Base'} T{offset+1}"), start=1):
        txt = generate_text(seed, use_will)
        q_new = embed(txt)
        lam = lambda_fn(step) * LAMBDA_AMP
        if use_will:
            q, p = leapfrog_will(q_new, p, q0, q_will, lam)
        else:
            q, p = leapfrog_base(q_new, p, q0, lam)
        if step % REFRESH_EVERY == 0:
            p = np.random.normal(scale=SIGMA_REFRESH, size=p.shape)
        records.append({
            "step": step,
            "drift": np.linalg.norm(q - q0),
            "pnorm": np.linalg.norm(p),
            "A_q": float((q0 / np.linalg.norm(q0)) @ q)
        })
    df = pd.DataFrame(records)
    df["will"] = use_will
    df["trial"] = offset + 1
    return df

# ---------------- メイン ----------------
def main():
    corpus = load_corpus()
    # サンプル出力
    print("\n>> Sample Baseline")
    for i, s in enumerate(corpus[:3], 1):
        print(f"[Base {i}] {generate_text(s, False)}")
    print("\n>> Sample WiLL")
    for i, s in enumerate(corpus[:3], 1):
        print(f"[WiLL {i}] {generate_text(s, True)}")

    # ABテスト
    all_df = []
    lambda_sched = lambda t: 1 + np.sin(2 * np.pi * t / 50)
    for use, label in [(False, "Baseline"), (True, "WiLL")]:
        for t in range(TRIALS):
            print(f"\n=== {label} Trial {t+1}/{TRIALS} ===")
            all_df.append(run_experiment(lambda_sched, use, t))
    df = pd.concat(all_df, ignore_index=True)

    # 要約統計量
    print("\n=== Summary Statistics (printf) ===")
    for use, label in [(False, "Baseline"), (True, "WiLL")]:
        sub = df[df.will == use]
        for metric in ["drift", "pnorm", "A_q"]:
            arr = sub[metric]
            print(
                f"printf: {label} {metric} "
                f"mean={arr.mean():.4f} std={arr.std():.4f} "
                f"min={arr.min():.4f} median={arr.median():.4f} max={arr.max():.4f}"
            )

    # 時系列プロット
    for metric in ["drift", "pnorm", "A_q"]:
        plt.figure(figsize=(6, 4))
        for use, label in [(False, "Base"), (True, "WiLL")]:
            sub = df[df.will == use]
            plt.plot(sub.step, sub[metric], alpha=0.5, label=label)
        plt.title(metric)
        plt.xlabel("step")
        plt.legend()
        plt.grid(True)
        path = OUTDIR / f"{metric}_timeseries2_GPT.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.show()
        if files:
            files.download(str(path))

    # 平均バーグラフ
    stats = df.groupby("will")[ ["drift", "pnorm", "A_q"] ].mean().rename({False: "Base", True: "WiLL"})
    stats.plot.bar(figsize=(8, 3))
    plt.title("Mean Metrics by Mode")
    plt.ylabel("Value")
    plt.grid(axis="y")
    bar_path = OUTDIR / "GPT_mean_metrics2.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.show()
    if files:
        files.download(str(bar_path))

    # CSV保存
    csv_path = OUTDIR / "GPT_metrics2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    if files:
        files.download(str(csv_path))

if __name__ == "__main__":
    main()

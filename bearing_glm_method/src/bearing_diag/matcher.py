"""包络谱谐波匹配与故障部位判别。

算法（docs/principles.md 第 9 章）：

1. 对每个候选故障特征频率 f_c，在 k·f_c（k=1..K，默认 5）附近
   ±max(2%, 1.5 根谱线) 窗内搜索最大峰；
2. 峰的突出度按**局部邻域中位数**定标（邻域宽 ±max(16%, 10 根谱线)）。
   不用全局中位数的原因：窄带噪声的包络谱随频率衰减，低频段整体抬高，
   全局基线会让低频候选（FTF/BSF 低次谐波）系统性虚高——这是包络谱
   分析的经典误报来源；真正的离散谱线高出其邻域，衰减背景则不会；
3. 谐波命中判据 p ≥ 5；得分 = Σ w_k·min(p_k/20, 1) / Σ w_k，w_k = 1/k
   （低次谐波权重更高、突出度封顶，兼顾"谐波族完整性"与"强度"）；
4. 混淆消解：若两候选频率成整数倍（如 BPFO = 滚动体数 × FTF，恒成立），
   且小频率自身基频命中强，则大频率的证据判定为小频率谐波的"继承"，
   打折处理——避免保持架故障被误报为外圈故障；
5. 得分最高且命中数 ≥2、得分 ≥0.15 者判为故障部位；结合次高得分给出置信度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fault_freqs import FAULT_NAMES_CN

# 判据参数（经验值，可调；文档第 9 章有敏感性讨论）
PROMINENCE_MIN = 5.0     # 谐波命中所需最小突出度（峰/局部邻域中位数）
PROMINENCE_SAT = 20.0    # 突出度饱和值，达到即计满分
TOL_REL = 0.02           # 谐波搜索窗相对宽度（覆盖 1~2% 滑移）
LOCAL_TOL_REL = 0.16     # 局部基线邻域相对宽度（搜索窗的 8 倍）
LOCAL_MIN_BINS = 10      # 局部邻域至少包含的谱线数（不足则放宽）
N_HARMONICS_DEFAULT = 5
MIN_SCORE = 0.15         # 判为故障的最低得分
MIN_HITS = 2             # 判为故障的最低谐波命中数
CONFIRM_CONF = 0.5       # 置信度 ≥ 此值判"明确"
SUSPECT_CONF = 0.25      # 置信度 ≥ 此值判"疑似"


@dataclass
class CandidateScore:
    fault: str
    freq: float
    score: float
    hits: int
    harmonics: list[dict] = field(default_factory=list)
    # harmonics: [{"k", "target_hz", "found_hz", "amp", "prominence", "hit"}, ...]
    detectable: bool = True
    confounded: bool = False  # 证据可能继承自其整数倍小频率（如 BPFO vs FTF）


@dataclass
class MatchResult:
    candidates: list[CandidateScore]      # 按得分降序
    verdict: str                          # 'normal' | 'BPFO' | 'BPFI' | 'BSF' | 'FTF'
    confidence: float                     # 0~1
    notes: list[str] = field(default_factory=list)

    @property
    def verdict_cn(self) -> str:
        if self.verdict == "normal":
            return "正常（未发现明确故障特征）"
        return FAULT_NAMES_CN[self.verdict]

    @property
    def level(self) -> str:
        if self.verdict == "normal":
            return "未见故障特征"
        if self.confidence >= CONFIRM_CONF:
            return "明确"
        return "疑似（建议人工复核包络谱）"

    def top(self) -> CandidateScore | None:
        return self.candidates[0] if self.candidates else None


def match_fault_frequencies(
    freqs: np.ndarray,
    amps: np.ndarray,
    fault_freqs: dict[str, float],
    n_harmonics: int = N_HARMONICS_DEFAULT,
) -> MatchResult:
    """在包络谱上匹配故障特征频率的谐波族。

    freqs/amps: 包络谱（来自 spectrum.envelope_spectrum）。
    fault_freqs: {故障类型: 特征频率 Hz}，来自 Bearing.fault_frequencies 或用户给定。
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    if freqs.size < 8:
        raise ValueError("包络谱点数过少（采集时长不足）")
    df = float(freqs[1] - freqs[0])
    fmax = float(freqs[-1])
    w = np.array([1.0 / k for k in range(1, n_harmonics + 1)])
    w_sum = float(w.sum())

    notes: list[str] = []
    results: list[CandidateScore] = []
    for fault, fc in sorted(fault_freqs.items(), key=lambda kv: kv[1]):
        cs = CandidateScore(fault=fault, freq=fc, score=0.0, hits=0)
        if fc < 2.5 * df:
            cs.detectable = False
            notes.append(
                f"{fault}={fc:.2f} Hz 低于频率分辨率可检测下限（需更长采集时长）"
            )
            results.append(cs)
            continue
        if fc > fmax:
            cs.detectable = False
            notes.append(f"{fault}={fc:.2f} Hz 超出包络谱分析范围（{fmax:.0f} Hz）")
            results.append(cs)
            continue

        score_acc = 0.0
        for i, k in enumerate(range(1, n_harmonics + 1)):
            target = k * fc
            if target > fmax + df:
                # 缩小权重分母：未覆盖到的谐波不计入满分基数
                w_sum_eff = float(w[: i].sum()) if i > 0 else np.finfo(float).eps
                break
            half_win = max(TOL_REL * target, 1.5 * df)
            m = (freqs >= target - half_win) & (freqs <= target + half_win)
            if not np.any(m):
                continue
            j = int(np.argmax(amps[m]))
            found_hz = float(freqs[m][j])
            amp = float(amps[m][j])
            # 局部邻域中位数作基线（抗包络谱整体衰减形状的误报）
            half_local = max(LOCAL_TOL_REL * target, LOCAL_MIN_BINS * df)
            ml = (freqs >= target - half_local) & (freqs <= target + half_local)
            if np.count_nonzero(ml) >= 5:
                local_med = float(np.median(amps[ml]))
            else:
                local_med = float(np.median(amps))
            local_med = max(local_med, np.finfo(float).eps)
            prom = amp / local_med
            hit = prom >= PROMINENCE_MIN
            if hit:
                cs.hits += 1
                score_acc += w[i] * min(prom / PROMINENCE_SAT, 1.0)
            cs.harmonics.append({
                "k": k, "target_hz": target, "found_hz": found_hz,
                "amp": amp, "prominence": prom, "hit": hit,
            })
        else:
            w_sum_eff = w_sum

        cs.score = float(min(score_acc / w_sum_eff, 1.0)) if w_sum_eff > 0 else 0.0
        results.append(cs)

    # 混淆消解：整数倍关系且小频率基频自身命中 → 大频率打折
    for big in results:
        if big.score <= 0:
            continue
        for small in results:
            if small is big or small.score <= 0 or small.freq <= 0:
                continue
            ratio = big.freq / small.freq
            m_int = round(ratio)
            if m_int >= 2 and abs(ratio - m_int) <= TOL_REL:
                k1 = next((h for h in small.harmonics if h["k"] == 1), None)
                if k1 is not None and k1["hit"]:
                    big.score *= 0.4
                    big.confounded = True
                    notes.append(
                        f"{big.fault} 与 {small.fault} 成 {m_int} 倍频关系，"
                        f"且 {small.fault} 基频命中，{big.fault} 证据已按继承谐波打折"
                    )

    results.sort(key=lambda c: c.score, reverse=True)
    best = results[0]
    second = results[1] if len(results) > 1 else None

    if best.score < MIN_SCORE or best.hits < MIN_HITS or not best.detectable:
        verdict = "normal"
        confidence = float(best.score)
        notes.append(
            "各候选频率谐波得分均低于判据阈值；注意这不能证明轴承健康——"
            "早期故障、强噪声或采集时长不足都可能掩盖特征，建议结合峭度与人工读谱"
        )
    else:
        verdict = best.fault
        ratio_pen = 1.0
        if second is not None and second.score > 0 and best.score > 0:
            ratio_pen = 1.0 - 0.5 * min(second.score / best.score, 1.0)
        confidence = float(best.score * ratio_pen)

    return MatchResult(candidates=results, verdict=verdict,
                       confidence=confidence, notes=notes)

# 02 故障振动信号的数学模型与仿真器

> 对应代码：`src/bearing_fault/simulator.py`（`BearingSimulator` 类）。
> 阅读本章前请确保已掌握第 01 章的特征频率（FTF/BSF/BPFO/BPFI）及其物理机理。

第 01 章回答了"冲击以什么频率出现"，本章回答一个更具体的问题：**加速度计测到的电压信号到底长什么样？** 我们采用 Randall 等人建立、并被后续大量文献沿用的经典模型——准周期冲击串激励结构共振，再叠加噪声。本项目没有采用昂贵的真实采集数据作为主数据源，而是用这个模型写了一个约 200 行的仿真器 `BearingSimulator`，后续每一章的信号处理算法（包络解调、阶次分析、时频分析、故障判别）都先在它生成的、地面真值完全已知的信号上验证。因此，读懂这个仿真器就是读懂全书的"实验台"。

---

## 2.1 Randall 冲击响应模型总览

模型把轴承故障振动信号写成三项的乘积结构之叠加（`simulator.py` 模块 docstring 中给出了同样形式）：

$$
x(t) = \underbrace{\sum_{k} A_k \cdot h(t - t_k)}_{\text{故障分量 } s(t)} + \underbrace{n(t)}_{\text{噪声}} \tag{式 2-1}
$$

各项的物理含义：

- **$t_k$ — 第 $k$ 次冲击的发生时刻**。滚动体每碾过缺陷一次，接触区就产生一次极短的力脉冲，近似为 Dirac $\delta$ 函数。名义上冲击等间隔出现，间隔为故障特征周期 $T_f = 1/f_f$（$f_f$ 是 BPFO/BPFI/BSF/FTF 之一，见第 01 章）；实际上因滚动体滑差，间隔存在 1–2% 的随机抖动（§2.2）。
- **$h(t)$ — 结构对单次冲击的响应**。力脉冲激发轴承-轴承座结构的固有共振，传感器看到的是一个频率为 $f_n$、按指数衰减的振荡（§2.3）。系统线性时不变的假设下，总响应就是各次冲击响应的叠加，即冲击串与 $h(t)$ 的**卷积**。
- **$A_k$ — 第 $k$ 次冲击的幅值**。缺陷相对于载荷区的位置随时间变化，冲击力随之被周期性调制（§2.4）。
- **$n(t)$ — 加性高斯白噪声**。涵盖电磁干扰、其他机械部件的振动背景、采集链噪声等一切"与故障无关"的成分，其强度用信噪比定量（§2.5）。

模型成立的两个关键近似，也是它的适用边界：

1. **线性叠加**：各次冲击响应互不干扰地相加。当冲击间隔大于共振响应的有效持续时间时这是好的近似；反之响应会堆叠（见 §2.3 末尾的算例）。
2. **单一共振模态**：真实结构有多个模态，但实验（如 CWRU 数据集）显示轴承故障能量通常集中激发某一个高频共振带，单自由度模型已足够复现诊断算法所依赖的信号结构。

### 代码对应

`BearingSimulator.signal()`（`simulator.py:81`）的实现就是式 2-1 的直译：

```python
spikes = np.zeros(n)
idx = np.clip((impact_times * self.fs).astype(int), 0, n - 1)
np.add.at(spikes, idx, impact_amps)      # Σ_k A_k·δ(t − t_k) 的离散采样
clean = np.convolve(spikes, self._impulse_response())[:n]   # 与 h(t) 卷积
x = clean + noise                        # 加噪声
```

即：先在时间轴上放置加权脉冲串 $\sum_k A_k\,\delta(t-t_k)$，再与离散冲击响应 $h[n]$ 做卷积得到无噪故障分量 `clean`，最后加上噪声。`return_components=True` 时可额外返回 `clean / noise / t / theta / fr_t / impact_times`，方便逐成分检查（后续各章的教程均依赖这一接口）。

---

## 2.2 准周期冲击串：滑差为何是"抖动"而非"偏移"

### 名义间隔

忽略滑差时，第 $k$ 次冲击时刻为

$$
t_k = t_0 + k\,T_f,\qquad T_f = \frac{1}{f_f} \tag{式 2-2}
$$

其中 $t_0$ 是随机初相（缺陷相对于 $t=0$ 的角位置未知），$f_f$ 为对应的故障特征频率。以 SKF 6205、1797 rpm（$f_r = 29.95$ Hz）为例（数值由 `Bearing.sk6205().characteristic_frequencies(29.95)` 算出）：

| 故障类型 | 特征频率 $f_f$ | 名义周期 $T_f$ |
|---|---|---|
| 外圈 (BPFO) | 107.36 Hz | 9.314 ms |
| 内圈 (BPFI) | 162.19 Hz | 6.166 ms |
| 滚动体 (BSF) | 70.58 Hz | 14.167 ms |
| 保持架 (FTF) | 11.93 Hz | 83.827 ms |

### 滑差的建模

第 01 章已解释滑差的物理来源：滚动体与滚道间存在微量相对滑动（典型 1–2%），缺陷实际被碾过的时刻围绕名义值随机涨落。仿真器把它建模为**逐冲击独立的间隔抖动**（`_impacts()`，`simulator.py:167`）：

$$
t_k = t_0 + k\,T_f + \varepsilon_k,\qquad \varepsilon_k \sim \mathcal{N}\!\big(0,\;(\sigma_{\text{slip}} T_f)^2\big) \tag{式 2-3}
$$

其中 $\sigma_{\text{slip}}$ 即构造参数 `slip`，缺省 `0.01`（1%）。BPFO 算例下抖动的时间标准差为 $0.01 \times 9.314\ \text{ms} \approx 0.093\ \text{ms}$——在 12 kHz 采样下约 1.1 个采样点，肉眼在波形上完全看不出，但它对频域的影响是决定性的。

> **实现细节**：代码实际上在**轴相位域**加抖动（见 §2.6），`jitter = rng.normal(0, slip, size) * d_theta`，即角域标准差为 $\sigma_{\text{slip}}\cdot\Delta\theta$；恒速时相位与时间只差一个线性因子 $2\pi f_r$，故时间域标准差恰好是 $\sigma_{\text{slip}} T_f$，与式 2-3 一致。

### 为什么必须叫"准周期"

若把滑差误建成一个固定的周期偏移（例如间隔恒为 $1.01\,T_f$），信号仍是严格周期信号，其傅里叶谱是间隔为 $f_f/1.01$ 的**离散线谱**。而式 2-3 中 $\varepsilon_k$ 逐次独立，意味着相位噪声会随 $k$ **随机游走进而累积**：第 $k$ 次冲击相对名义相位的偏差随时间越来越大。其谱后果是：

- 故障频率的**低次谐波**仍是清晰的谱线（相位漂移尚小）；
- **高次谐波**逐渐展宽、抬高噪声底，乃至消失——真实轴承谱中"BPFO 前几次谐波清晰、高次模糊"正是这一现象。

这也是为什么第 04 章的包络解调要看**包络谱的低频端**而非原始谱的高频端。我们用 `seed=0` 的仿真验证：外圈故障 5 s 内共 537 次冲击（理论值 $5 \times 107.36 \approx 536.8$），相邻间隔的相对标准差实测约 1.45%。

> 注意 1.45% ≈ $\sqrt{2}\times 1\%$ 并非笔误：`slip` 控制的是**每次冲击绝对时刻**的抖动标准差；而相邻间隔 $t_{k+1}-t_k$ 的抖动是两次独立抖动的差，方差相加，故标准差放大 $\sqrt{2}$ 倍。

---

## 2.3 结构共振响应：单自由度欠阻尼系统

### 从运动方程到 h(t)

把轴承-轴承座结构在共振频率附近近似为单自由度质量-弹簧-阻尼系统。冲击力脉冲 $F\cdot\delta(t)$ 作用下，位移响应满足

$$
m\ddot{y} + c\dot{y} + k_s y = F\,\delta(t) \tag{式 2-4}
$$

引入固有频率 $f_n = \frac{1}{2\pi}\sqrt{k_s/m}$ 与阻尼比 $\zeta = \frac{c}{2\sqrt{k_s m}}$。欠阻尼（$\zeta \ll 1$）时，式 2-4 的解是衰减振荡；忽略 $\mathcal{O}(\zeta^2)$ 小量（有阻尼频率 $\omega_d = \omega_n\sqrt{1-\zeta^2} \approx \omega_n$）并吸收幅度常数后，仿真器采用（`_impulse_response()`，`simulator.py:183`）：

$$
h(t) = e^{-2\pi\zeta f_n t}\,\sin(2\pi f_n t),\qquad t \ge 0 \tag{式 2-5}
$$

各项物理意义：

- $\sin(2\pi f_n t)$：响应以结构共振频率 $f_n$ 振荡——这就是为什么故障能量从低频的故障频率"搬移"到了高频共振带，也是第 04 章共振解调的基本出发点；
- $e^{-t/\tau}$：**指数衰减包络**，时间常数

$$
\tau = \frac{1}{2\pi\zeta f_n} \tag{式 2-6}
$$

表示振荡幅度衰减到初值 $1/e \approx 37\%$ 所需的时间。$\zeta$ 越小、$f_n$ 越低，振铃拖得越久。

### 数值算例

取缺省参数 `resonance_freq` $f_n = 4000$ Hz、`damping` $\zeta = 0.05$：

$$
\tau = \frac{1}{2\pi \times 0.05 \times 4000} \approx 0.796\ \text{ms}
$$

即每次冲击激起一段 4 kHz 的"振铃"，约 0.8 ms 后衰减到 37%。代码把 $h[n]$ 截断到约 6 个时间常数（$e^{-6} \approx 0.25\%$，可忽略）：`n_h = max(8, int(6·tau·fs))`，代入 $f_s = 12$ kHz 得 $6\tau \approx 4.77$ ms、共 **57 个采样点**。

把两个时间尺度放在一起看：BPFO 周期 9.31 ms > $6\tau$ ≈ 4.77 ms，外圈故障的相邻振铃基本分离；而 BPFI 周期 6.17 ms 已接近 $6\tau$，内圈故障的振铃拖尾会与下一次冲击的响应部分叠加——这正是式 2-1 线性叠加假设被"压到边界"的情形，仿真信号仍忠实反映了这一真实效应。

> **构造约束**：`__init__` 中要求 `resonance_freq < fs/2`（奈奎斯特频率），否则直接抛 `ValueError`——4 kHz 的共振用 12 kHz 采样（奈奎斯特 6 kHz）是安全的。

---

## 2.4 幅值调制：缺陷相对载荷区的运动

式 2-1 中的 $A_k$ 并非常数。轴承载荷（径向力）在空间中是**定向**的，存在一个"载荷区"；缺陷每次进入载荷区时冲击最强，转出时最弱。缺陷是否相对于载荷区运动，决定了三种故障截然不同的调制模式（`_impacts()`，`simulator.py:173-180`）：

**内圈故障——按转频 $f_r$ 调制。** 内圈随轴旋转，缺陷每转一周进出载荷区一次，调制周期等于轴转周期。以轴相位 $\theta$（单位 rad，每转 $2\pi$）为自变量：

$$
A_k^{\text{(inner)}} = 1 + m\cos\theta_k \tag{式 2-7}
$$

**滚动体故障——按保持架频率 FTF 调制。** 滚动体上的缺陷随保持架公转，进出载荷区的速率是保持架转速。保持架相对轴的角速度之比恰为 FTF 的阶次比 $0.3983$（即 $\frac{1}{2}(1-\frac{d}{D}\cos\alpha)$，见第 01 章），故

$$
A_k^{\text{(ball)}} = 1 + m\cos\!\big(0.3983\cdot\theta_k\big) \tag{式 2-8}
$$

代码中这个系数写作 `ftf_order = 0.5 * (1.0 - bearing.geometry_ratio)`，不随转速变化。

**外圈故障——无调制。** 外圈固定，缺陷相对载荷区的位置不变（通常在载荷区内），冲击力近似恒定：

$$
A_k^{\text{(outer)}} = 1 \tag{式 2-9}
$$

（保持架故障同样不做调制，按常数幅值处理。）

其中 $m$ 是构造参数 `modulation_depth`，缺省 **0.6**，幅值在 $[1-m,\,1+m] = [0.4,\,1.6]$ 内摆动。

**调制的谱后果**（第 04 章将定量展开）：幅值被频率 $f_m$ 的余弦调制后，频谱以故障频率为中心分裂出间隔 $f_m$ 的**边带**。因此内圈故障的谱峰两侧出现 $\pm f_r$ 边带、滚动体故障出现 $\pm$FTF 边带，而外圈故障是"干净"的单线谱——边带模式本身就是区分故障位置的证据。调制现象在代码中只改 $A_k$ 三个分支，但其诊断价值贯穿后续所有章节。

---

## 2.5 加性高斯白噪声与信噪比

模型的最后一项 $n(t)$ 取**加性高斯白噪声**（AWGN）：各采样点独立、同分布、零均值高斯，功率谱在所有频率上平坦（`_make_noise()`，`simulator.py:190`）。噪声强度不直接给幅值，而是以**信噪比**定量：

$$
\mathrm{SNR_{dB}} = 10\,\log_{10}\frac{P_{\text{sig}}}{P_{\text{noise}}},\qquad
P_{\text{sig}} = \frac{1}{N}\sum_{n} s[n]^2 \tag{式 2-10}
$$

$P_{\text{sig}}$ 取无噪故障分量 `clean` 的均方值（对零均值或近零均值信号即方差/平均功率）。代码由目标 `snr_db` 反解噪声功率并生成噪声：

```python
p_noise = p_sig / (10.0 ** (snr_db / 10.0))
noise   = np.sqrt(p_noise) * rng.standard_normal(n)
```

**负 dB 的含义**：对数尺度下，$\mathrm{SNR_{dB}}<0$ 表示 $P_{\text{sig}} < P_{\text{noise}}$——**噪声功率大于信号功率**。`signal()` 的缺省 `snr_db = -6.0` 对应

$$
\frac{P_{\text{noise}}}{P_{\text{sig}}} = 10^{0.6} \approx 3.98
$$

即噪声功率约为故障分量的 **4 倍**（噪声标准差约为其 2 倍）。实测验证（`seed=0`、外圈故障）：合成信号中按式 2-10 重算的 SNR 为 $-5.9999$ dB，与目标一致；噪声标准差 0.290，与反解值相符。这并非刁难读者：工业现场早期故障的冲击分量几乎总是埋在背景振动之下，"信号比噪声弱"恰是常态，也是后续章节所有增强算法存在的理由。设 `snr_db=None` 则完全不加噪声，用于观察理想故障分量。

**退化情形**：健康轴承（`fault="healthy"`）没有故障分量，$P_{\text{sig}}=0$，式 2-10 无意义。代码此时退化为标准差 **0.1** 的固定低幅值白噪声（与 `snr_db` 无关），作为"正常背景振动"的替身。

---

## 2.6 变速工况：以轴相位为主变量

### 为什么不能用时间做等间隔采样冲击

转速变化时，故障特征频率与转频同步变化（阶次比 $f_f/f_r$ 是常数，只取决于几何，见第 01 章），冲击在**时间**上不再等间隔；但它们在**轴转角**上仍然等间隔——缺陷每相对滚动体转过固定角度 $\Delta\theta = 2\pi/\text{order}$ 就被碾过一次。因此正确的做法是：以轴相位 $\theta(t)$ 为主变量生成冲击，再映射回时间轴。

轴相位定义为瞬时转频 $f_r(t)$（Hz，即每秒转数）的积分：

$$
\theta(t) = 2\pi\int_0^t f_r(u)\,\mathrm{d}u \tag{式 2-11}
$$

$\theta(t)$ 单调递增、可逆，因此 $t \leftrightarrow \theta$ 的映射一一对应。冲击在相位域生成：

$$
\theta_k = \theta_0 + k\,\Delta\theta + \varepsilon_k,\qquad
\Delta\theta = \frac{2\pi}{\text{order}} \tag{式 2-12}
$$

（抖动 $\varepsilon_k$ 同 §2.2，角域标准差 $\sigma_{\text{slip}}\Delta\theta$。）再用 $\theta(t)$ 的反函数插值回时间：代码中即 `imp_t = np.interp(imp_theta, theta, t)`。同一套逻辑天然覆盖恒速与变速。

### 匀变速下 θ(t) 的推导

`speed_ramp=(rpm_0, rpm_1)` 模拟匀变速：转速在时长 $T$ 内从 $f_0 = \mathrm{rpm}_0/60$ 线性变到 $f_1 = \mathrm{rpm}_1/60$，即

$$
f_r(t) = f_0 + \dot{f}\,t,\qquad \dot{f} = \frac{f_1 - f_0}{T} \tag{式 2-13}
$$

代入式 2-11 积分：

$$
\theta(t) = 2\pi\int_0^t (f_0 + \dot{f}\,u)\,\mathrm{d}u
          = 2\pi\left(f_0\,t + \tfrac{1}{2}\dot{f}\,t^2\right) \tag{式 2-14}
$$

与 `simulator.py:119` 逐符号一致（`rate` 即 $\dot f$）。恒速工况只是 $\dot f = 0$ 的特例：$\theta(t) = 2\pi f_r t$。

**算例**：`speed_ramp=(1200, 2400)`、5 s，则 $f_0 = 20$ Hz、$f_1 = 40$ Hz、$\dot f = 4$ Hz/s。我们实测验证了三点：(i) 仿真器输出的 `theta` 与式 2-14 逐点相符；(ii) 把 `impact_times` 用 `theta(t)` 变换到角域后，相邻冲击的角间隔均值恰为 $\Delta\theta = 1.7527$ rad（外圈，order = 3.5848），且除滑差抖动外严格等间隔；(iii) 时间域间隔则随转速上升而缩短——等角度、不等时间。若此时直接对时间信号做 FFT，故障谱峰会因频率随时间漂移而**涂抹**成宽峰，这就是第 05 章要解决的问题；仿真器随信号一并输出 `theta` 与 `fr_t`，正是为阶次分析的角域重采样提供依据。

---

## 2.7 参数对应表

`BearingSimulator.__init__` 的每个构造参数与本章符号的对应关系：

| 构造参数 | 缺省值 | 模型符号 | 含义 | 出现位置 |
|---|---|---|---|---|
| `bearing` | `Bearing.sk6205()` | $n, d, D, \alpha$ | 轴承几何，决定特征频率与阶次比 | §2.2、式 2-12 |
| `fs` | `12000.0` | $f_s$ | 采样频率 (Hz) | 全文 |
| `duration` | `5.0` | $T$ | 信号时长 (s) | §2.6 |
| `rpm` | `1797.0` | $f_r = \mathrm{rpm}/60$ | 轴转速；恒速时转频 29.95 Hz | §2.4、§2.6 |
| `resonance_freq` | `4000.0` | $f_n$ | 结构共振频率，须 $< f_s/2$ | 式 2-5 |
| `damping` | `0.05` | $\zeta$ | 共振阻尼比 | 式 2-5、2-6 |
| `slip` | `0.01` | $\sigma_{\text{slip}}$ | 冲击间隔相对抖动标准差（滑差） | 式 2-3、2-12 |
| `modulation_depth` | `0.6` | $m$ | 幅值调制深度，$A_k\in[1-m,1+m]$ | 式 2-7、2-8 |
| `seed` | `None` | — | 随机种子，固定后信号可复现 | — |

`signal()` 方法自身的参数：

| 方法参数 | 缺省值 | 含义 |
|---|---|---|
| `fault` | `"healthy"` | 故障类型：`healthy / inner / outer / ball / cage` |
| `snr_db` | `-6.0` | 信噪比 (dB)，式 2-10；`None` 不加噪声 |
| `speed_ramp` | `None` | `(起始 rpm, 结束 rpm)`，给出时按式 2-13/2-14 匀变速 |
| `return_components` | `False` | `True` 返回 `signal/clean/noise/t/theta/fr_t/impact_times` 字典 |

---

## 本章小结

- 故障信号 = 准周期冲击串 $\otimes$ 结构共振响应 $+$ 高斯白噪声（式 2-1）；
- 冲击间隔名义值由特征频率决定，1–2% 的滑差抖动使信号"准"周期而非严格周期，表现为高次谐波展宽；
- 共振响应是指数衰减振荡，缺省参数下 $\tau \approx 0.8$ ms、截断长度 57 点；
- 幅值调制的有无与频率（$f_r$ / FTF / 无）取决于故障位置，产生诊断可用的边带模式；
- 缺省 SNR = −6 dB，噪声功率约为信号 4 倍；
- 变速时以轴相位 $\theta(t)$ 为主变量保证冲击等角度分布，相位信号同时供第 05 章阶次分析使用。

至此我们有了"已知答案的考题"。下一章（第 03 章）先不动用频域，看看最简单的时域统计指标能在这台"实验台"上读出什么。

---

## 参考文献

1. R. B. Randall, J. Antoni, "Rolling element bearing diagnostics—A tutorial," *Mechanical Systems and Signal Processing*, 2011. —— 本章模型的总体框架与边带分析的经典综述。
2. P. D. McFadden, J. D. Smith, "Model for the vibration produced by a single point defect in a rolling element bearing," *Journal of Sound and Vibration*, 1984. —— 单点缺陷冲击响应模型的奠基性工作。
3. P. D. McFadden, J. D. Smith, "The vibration produced by multiple point defects in a rolling element bearing," *Journal of Sound and Vibration*, 1985. —— 多缺陷与幅值调制的推广。
4. R. B. Randall, *Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications*, Wiley, 2011. —— 工程背景与滑差、共振解调等概念的系统论述。
5. J. Antoni, "The spectral kurtosis: a useful tool for characterising non-stationary signals," *Mechanical Systems and Signal Processing*, 2006. —— 冲击性信号的统计刻画，与第 03 章的峭度指标呼应。

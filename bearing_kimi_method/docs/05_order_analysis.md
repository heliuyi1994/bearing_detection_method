# 05 阶次分析：变转速工况的角域重采样

第 04 章的包络分析有一个隐含前提：转频 $f_r$ 恒定，故障特征频率 $f_{\text{BPFO}}$ 等
是一条不随时间移动的谱线。实际工况中轴承经常运行在升速、降速或负载波动状态，
此时直接对等时间采样信号做 FFT，谱峰会被"涂抹"（frequency smearing）成一滩
宽峰，无法与理论特征频率比对。**阶次分析**（order analysis）把信号从等时间域
变换到等角度域再做谱分析，使谱峰位置只取决于轴承几何而与转速无关。

本章对应 `src/bearing_fault/order_analysis.py`，相位 $\theta(t)$ 的生成见
`src/bearing_fault/simulator.py`。

## 5.1 问题：变转速下的频率涂抹

### 5.1.1 冲击"等角度"而非"等时间"出现

由第 01 章的特征频率推导，外圈滚道上两个相邻滚动体之间的圆心角是固定的
$2\pi/n$。轴每转过一个固定角度，缺陷就被碾过一次——**冲击在角度域等间隔**，
间隔为

$$
\Delta\theta_{\text{imp}} = \frac{2\pi}{O} \quad\text{（弧度）},
\tag{5.1}
$$

其中 $O$ 是该故障相对转频的阶次（见 5.2 节）。只有恒速时"等角度"才等价于
"等时间"：$\Delta t = \Delta\theta / (2\pi f_r)$。转速变化时，冲击的瞬时重复
频率跟随转频：

$$
f_{\text{fault}}(t) = O \cdot f_r(t).
\tag{5.2}
$$

仿真器正是按这个物理事实实现的：`_impacts()`（`simulator.py:146`）先在轴相位
域以等角度间隔 $\Delta\theta = 2\pi/O$ 布置冲击，再换算回时间域，因此同一套
代码同时覆盖恒速与变速。

### 5.1.2 算例：匀变速 1200 → 1800 rpm 的涂抹量

设轴在 $T=4$ s 内从 1200 rpm 匀加速到 1800 rpm，瞬时转频

$$
f_r(t) = f_{r0} + a\,t, \qquad
f_{r0} = 20\ \text{Hz},\quad
a = \frac{30-20}{4} = 2.5\ \text{Hz/s}.
\tag{5.3}
$$

SKF 6205 外圈故障阶次 $O_{\text{BPFO}} = 3.5848$（第 01 章），由式 (5.2)：

$$
f_{\text{BPFO}}(t):\quad 3.5848 \times 20 = 71.70\ \text{Hz}
\ \longrightarrow\ 3.5848 \times 30 = 107.54\ \text{Hz}.
$$

也就是说，在 4 s 的采集时间里，故障谱线从 71.7 Hz 匀速漂移到 107.5 Hz，
漂移总量

$$
\Delta f = O \cdot \Delta f_r = 3.5848 \times 10 \approx 35.8\ \text{Hz}.
\tag{5.4}
$$

而 4 s 信号的频率分辨率只有 $1/T = 0.25$ Hz——谱线能量被均匀地摊在约
$35.8/0.25 \approx 143$ 个谱线上。恒速时收敛成一根细线的能量，现在铺成
几十 Hz 宽的矮丘，这就是频率涂抹。

### 5.1.3 实测涂抹效果

用本项目代码复现（外圈故障，`snr_db=10`，带通 2000–5400 Hz 后取 Hilbert
包络，流程见 5.5 节）：

```python
sim = BearingSimulator(fs=12000, duration=4.0, rpm=1200, seed=4)
comp = sim.signal("outer", snr_db=10.0, speed_ramp=(1200, 1800),
                  return_components=True)
env = analytic_envelope(bandpass(comp["signal"], 12000, 2000, 5400))
freqs, amps = envelope_spectrum(env, 12000)   # 第 04 章的等时间包络谱
```

等时间包络谱在 20–300 Hz 内的主峰落在 **90.5 Hz** 附近——恰是漂移中点
$f_{\text{BPFO}}(25\ \text{Hz}) = 89.6$ Hz 附近，但半功率宽度约 **12.8 Hz**，
峰顶平坦、两侧无明确边界。这样的谱既不能判定"是不是 3.5848 阶对应的频率"，
也无法与谐波、边带区分。**结论：转速变化超过谱线宽度时，基于等时间采样的
频谱判读失效**，必须换到角度域。

## 5.2 阶次：与转速无关的故障指纹

**定义（阶次，order）**：某频率成分与轴转频之比

$$
O = \frac{f}{f_r} \qquad\text{（单位：阶，或"次/转"）}.
\tag{5.5}
$$

物理意义：轴每转一圈，该事件发生的次数。由第 01 章的特征频率公式，四个故障
频率都正比于 $f_r$，比例系数只含几何量，因此**故障阶次是常数，与转速无关**：

$$
\begin{aligned}
O_{\text{FTF}}  &= \tfrac{1}{2}\left(1 - \tfrac{d}{D}\cos\alpha\right), &
O_{\text{BSF}}  &= \tfrac{D}{2d}\left(1 - \left(\tfrac{d}{D}\cos\alpha\right)^2\right),\\
O_{\text{BPFO}} &= \tfrac{n}{2}\left(1 - \tfrac{d}{D}\cos\alpha\right), &
O_{\text{BPFI}} &= \tfrac{n}{2}\left(1 + \tfrac{d}{D}\cos\alpha\right).
\end{aligned}
\tag{5.6}
$$

SKF 6205（$n=9$，$d=7.94$ mm，$D=39.04$ mm，$\alpha=0°$，几何因子
$(d/D)\cos\alpha = 0.20338$，见 `bearing.py:41`）代入得：

| 故障类型 | 阶次 $O$ | 1797 rpm 下频率 (Hz) | 1200→1800 rpm 漂移范围 (Hz) |
|---|---|---|---|
| 保持架 FTF | 0.3983 | 11.93 | 7.97 – 11.95 |
| 滚动体 BSF | 2.3567 | 70.58 | 47.13 – 70.70 |
| 外圈 BPFO  | 3.5848 | 107.36 | 71.70 – 107.54 |
| 内圈 BPFI  | 5.4152 | 162.19 | 108.30 – 162.46 |

代码里这个"与转速无关"体现在 `Bearing.characteristic_frequencies(1.0)`：
令 $f_r=1$ Hz 直接读出阶次。测试 `tests/test_order_analysis.py:18` 正是用
`B.bpfo(1.0)` 取外圈理论阶次 3.5848。诊断时只要在**阶次谱**上寻找这些常数
位置的谱峰即可，完全不需要知道采集期间的瞬时转速。

与阶次相对的概念是**频率固定成分**（如电网激励、共振频率）：它们在角度域
反而会被涂抹。阶次分析专门服务于"与轴同步"的成分。

## 5.3 角域重采样

### 5.3.1 轴相位 $\theta(t)$

定义轴累计转过的角度（弧度）为转频的积分：

$$
\theta(t) = 2\pi \int_0^t f_r(\tau)\,d\tau .
\tag{5.7}
$$

恒速时 $\theta(t) = 2\pi f_r t$ 是线性函数；匀变速（式 5.3）时

$$
\theta(t) = 2\pi\left(f_{r0}\,t + \tfrac{1}{2} a t^2\right),
\tag{5.8}
$$

即 `simulator.py:119` 的实现。只要转速不为零且不变向，$f_r(t)>0$ 保证
$\theta(t)$ **严格单调递增，从而可逆**——这是整个方法的数学基础。
`angular_resample()` 的第一步就是检查这个前提
（`order_analysis.py:32`，非单调抛出 `ValueError`）。

对上面的算例，4 s 内总转数

$$
N_{\text{rev}} = \frac{\theta(T)-\theta(0)}{2\pi}
= f_{r0}T + \tfrac{1}{2}aT^2 = 20\times4 + \tfrac{1}{2}\times2.5\times16
= 100\ \text{转}.
\tag{5.9}
$$

### 5.3.2 等角度采样：求逆映射 $t_k = \theta^{-1}(\theta_k)$

目标是得到等角度间隔的采样序列。取角度步长

$$
\Delta\theta = \frac{2\pi}{N_r},
\tag{5.10}
$$

其中 $N_r$ 是**每转采样点数**（`samples_per_rev`，默认 512），则等角度采样
节点为

$$
\theta_k = \theta_0 + k\,\Delta\theta, \qquad k = 0, 1, \dots, N_{\text{out}}-1,
\qquad N_{\text{out}} = \lfloor N_{\text{rev}}\cdot N_r \rfloor .
\tag{5.11}
$$

这些角度对应的**时间**由逆映射给出：

$$
t_k = \theta^{-1}(\theta_k),
\tag{5.12}
$$

再对原信号取值 $x_k = x(t_k)$。$\{x_k\}$ 就是角域信号：相邻样本相差
$1/N_r$ 转，原本"等角度出现"的冲击串在其中变成严格的等间隔序列。

对匀变速（式 5.8），逆映射可以解析写出（解关于 $t$ 的二次方程取正根）：

$$
t_k = \frac{-f_{r0} + \sqrt{f_{r0}^2 + 2a\,\theta_k/(2\pi)}}{a}.
\tag{5.13}
$$

但一般工况下 $\theta(t)$ 来自实测、没有解析式，只能用**数值求逆**：已知
采样表 $(t_i, \theta_i)$，对给定 $\theta_k$ 在表中反查 $t_k$——这恰好是
一维插值。再查表 $(t_i, x_i)$ 得 $x_k$。两次查表可以用一次插值合并：

$$
x_k = \mathrm{interp}\big(\theta_k;\ (\theta_i, x_i)\big),
\tag{5.14}
$$

即以 $\theta$ 为自变量直接插值 $x$。

### 5.3.3 本项目实现：线性插值 `np.interp`

`angular_resample()`（`order_analysis.py:15`）完整实现了上面的推导：

```python
n_rev = (theta[-1] - theta[0]) / (2.0 * np.pi)      # 式 (5.9)
n_out = int(n_rev * samples_per_rev)                # 式 (5.11)
theta_u = theta[0] + np.arange(n_out) * (2.0 * np.pi / samples_per_rev)
x_u = np.interp(theta_u, theta, x)                  # 式 (5.14)
```

`np.interp` 是分段线性插值：对每个 $\theta_k$ 找到包含它的区间
$[\theta_i, \theta_{i+1}]$，在两点间直线取值。它一步完成了"反查时间 +
取信号值"（式 5.14），无需显式构造 $t_k$。对算例信号验证：用
`np.interp(θ_k, theta, t)` 反查的时间与解析逆（式 5.13）的最大误差约
$1\times10^{-10}$ s，远小于采样间隔 $1/f_s \approx 83\ \mu$s——相位表的
时间轴足够密时，线性插值的求逆误差可忽略。

两个防御性检查值得注意：$\theta$ 必须严格单调（否则逆映射不存在）；总转数
太少（$N_{\text{out}} < 8$）时拒绝重采样——圈数太少连一个谱线都无法形成。

### 5.3.4 每转采样点数与阶次奈奎斯特

角域序列的"采样率"是 $N_r$ 点/转，由奈奎斯特定理，可表示的最高阶次为

$$
O_{\text{Nyq}} = \frac{N_r}{2}.
\tag{5.15}
$$

默认 $N_r = 512$，$O_{\text{Nyq}} = 256$ 阶。选择 $N_r$ 要兼顾两端：

- **下限**：$N_r/2$ 必须覆盖最高关注阶次。轴承故障阶次都在个位数
  （SKF 6205 最高 BPFI = 5.4152 阶），即使留几十次谐波裕量，$N_r = 64$
  都够；512 非常宽裕。
- **上限**：角域采样密度不应显著超过原时间采样密度，否则插值在"凭空造点"。
  转速 $f_r$ 时每转有 $f_s/f_r$ 个时间采样点，故宜
  $N_r \lesssim f_s / f_{r,\max}$。本例 $12000/30 = 400$ 点/转，
  $N_r = 512$ 略密，但线性插值的平滑作用只影响数百阶以外，对个位数阶次的
  故障判读无影响。

## 5.4 阶次谱

### 5.4.1 定义与实现

对角域序列做 DFT，即得**阶次谱**——横轴是"每转次数"（阶）。`order_spectrum()`
（`order_analysis.py:43`）的流程：

1. 调 `angular_resample()` 得角域序列 $x_u$；
2. 去均值 $x_u \leftarrow x_u - \bar{x}_u$；
3. 加 Hann 窗 $w_k$ 抑制频谱泄漏（与第 04 章 `spectrum.py` 的约定一致），
   幅值按窗增益 $G = \sum_k w_k$ 修正；
4. 实信号 FFT 并换算成单边幅值谱：

$$
S[m] = \frac{2}{G}\left|\sum_{k=0}^{N-1} x_u[k]\,w_k\,
e^{-j2\pi mk/N}\right|, \qquad S[0]\ \text{减半}.
\tag{5.16}
$$

5. 横轴用 `np.fft.rfftfreq(n, d=1/samples_per_rev)` 生成：把"采样间隔"
   设为 $1/N_r$ 转，DFT 频率轴的单位自然就是**次/转 = 阶**：

$$
O_m = \frac{m \cdot N_r}{N}, \qquad m = 0, 1, \dots, N/2.
\tag{5.17}
$$

**阶次分辨率**：序列总长 $N = N_{\text{rev}}\cdot N_r$ 点，故

$$
\Delta O = \frac{N_r}{N} = \frac{1}{N_{\text{rev}}} \quad\text{（阶）}.
\tag{5.18}
$$

分辨率只由**总转数**决定，与采样率、时长无关：100 转（算例）给
$\Delta O = 0.01$ 阶。要分清相距 1 阶的调制边带（如内圈故障的
$O_{\text{BPFI}} \pm 1$），理论上几圈就够；但要把谱峰定位到 2% 以内
（如诊断得分所需），几十到上百圈是实用的量级。

### 5.4.2 算例验证

对 5.1.3 节同一段被涂抹到无法判读的变速信号，继续做：

```python
orders, spec = order_spectrum(env, comp["theta"])   # 默认 samples_per_rev=512
```

结果：$O > 0.5$ 范围内主峰位于 **3.58 阶**，理论值 3.5848 阶，偏差 0.13%
（测试要求 ±2% 以内，见 `tests/test_order_analysis.py:50`）。涂抹成一滩的
能量在角度域重新聚拢成一根细线——因为故障冲击在角度域严格等间隔
（式 5.1），角域序列近似以 $2\pi/O$ 为周期，DFT 自然在 $O$ 及其谐波处
出现谱线。同一信号、两种谱：时域包络谱峰 90.5 Hz、宽 12.8 Hz（不可判读）；
阶次谱峰 3.58 阶、宽度受分辨率 0.01 阶限制（可判读）。这就是阶次分析的价值。

## 5.5 完整分析流程

变速工况的诊断流程是在恒速流程（第 04 章）中间插入角域重采样：

```
原始信号 x(t), 相位 θ(t)
   │  ① 带通滤波 bandpass(x, fs, 2000, 5400)      —— 选出共振频带（第 04 章）
   ▼  ② Hilbert 包络 analytic_envelope(x)         —— 解调出故障重复率（第 04 章）
包络 env(t)
   │  ③ 角域重采样 angular_resample(env, theta, 512)
   ▼
等角度序列 env(θ)
   │  ④ 阶次谱 order_spectrum(...)（内部含③）
   ▼
在 O_FTF / O_BSF / O_BPFO / O_BPFI 处找峰，与式 (5.6) 理论阶次比对（第 07 章）
```

与测试 `test_order_analysis.py` 中完全一致的代码：

```python
from bearing_fault import (Bearing, BearingSimulator, bandpass,
                           analytic_envelope, order_spectrum)

B = Bearing.sk6205()
sim = BearingSimulator(fs=12000, duration=4.0, rpm=1200, seed=4)
comp = sim.signal("outer", snr_db=10.0, speed_ramp=(1200, 1800),
                  return_components=True)

env = analytic_envelope(bandpass(comp["signal"], 12000, 2000, 5400))
orders, amps = order_spectrum(env, comp["theta"])
peak = orders[orders > 0.5][np.argmax(amps[orders > 0.5])]
assert abs(peak - B.bpfo(1.0)) / B.bpfo(1.0) < 0.02   # 命中 3.5848 阶
```

两个细节：

- **包络必须在重采样之前提取**。共振载波 $f_n = 4000$ Hz 对应 133–200 阶
  （随转速变化），在角度域它也是涂抹的；先解调把载波去掉，留下的包络只含
  低阶成分，重采样后谱才干净。
- 必须先有与 $x$ **等长、同采样时刻**的相位序列 $\theta(t)$。仿真器通过
  `return_components=True` 直接输出（`simulator.py:140`）；实测中的来源见
  下一节。

## 5.6 工程注意事项

1. **相位/转速信号的准确性是命脉**。阶次谱把一切误差都转嫁到 $\theta(t)$
   上：若相位有相对误差 $\delta\theta$，谱峰在阶次轴上同样被涂抹。实测中
   $\theta(t)$ 通常来自键相传感器（每转一个脉冲）或高线数编码器，再对脉冲
   时刻拟合平滑的 $\theta(t)$。本项目属简化情形：$\theta(t)$ 由仿真器按
   式 (5.8) 精确给出，反查误差 $10^{-10}$ s 量级，可视为理想相位。

2. **与计算阶次跟踪（COT）的关系**。工业界的 computed order tracking 正是
   本章方法：由转速计脉冲序列拟合 $\theta(t)$（常用键相脉冲 + 三次样条），
   再按式 (5.11)–(5.14) 重采样。COT 的难点在脉冲稀疏时的相位拟合精度；
   本项目假设 $\theta(t)$ 已知且精确，是 COT 的"理想相位"简化版，把重点放在
   重采样与阶次谱本身。无转速计时的替代方案是从信号中提取瞬时转频
   （如时频脊线，见第 06 章），积分得 $\theta(t)$，称无键相阶次跟踪。

3. **插值误差**。线性插值（`np.interp`）相当于对信号加了随频率恶化的低通
   （其频率响应为 $\mathrm{sinc}^2$ 形），高阶成分幅值被衰减、并可能产生
   插值混叠。对低阶故障成分（个位数阶）无实际影响；若需分析高阶次或精确
   幅值，应改用样条/带限插值，或提高原始 $f_s$ 使每转时间采样点充足。

4. **单调性与变向**。方法要求 $\theta(t)$ 严格单调递增（代码强制检查）。
   停机变向、转速过零的工况不能直接处理，需分段。此外分析默认一次采集内
   转速平滑变化；剧烈抖动（转速计的量化噪声、扭振）会污染 $\theta(t)$，
   应先对转速信号平滑。

5. **窗与泄漏的约定与第 04 章一致**：去均值、Hann 窗、幅值修正 $2/G$，
   因此阶次谱幅值可与包络谱幅值直接对比；代价是谱线主瓣宽约 $\pm 2\Delta O$，
   分辨极接近的阶次（如滑差造成的谱线簇）时需要更多转数。

6. **适用范围提醒**：阶次分析只对"与轴同步"的成分有效。固定频率成分
   （共振、电磁激励）在阶次谱上反而涂抹；两者应与时域包络谱（第 04 章）
   配合使用，而非互相替代。

## 参考文献

1. R. B. Randall, J. Antoni, "Rolling element bearing diagnostics—A tutorial,"
   *Mechanical Systems and Signal Processing*, 2011.（阶次分析与包络分析的
   经典综述）
2. K. R. Fyfe, E. D. S. Munck, "Analysis of computed order tracking,"
   *Mechanical Systems and Signal Processing*, 1997.（COT 的误差分析：
   相位拟合、插值与混叠）
3. K. M. Bossley, R. J. McKendrick, C. J. Harris, C. Mercer, "Hybrid computed
   order tracking," *Mechanical Systems and Signal Processing*, 1999.（转速计
   与信号联合的改进 COT）
4. P. D. McFadden, J. D. Smith, "Model for the vibration produced by a single
   point defect in rolling element bearing under radial load," *Journal of
   Sound and Vibration*, 1984.（故障冲击模型，见第 02 章）
5. P. D. McFadden, J. D. Smith, "The vibration produced by multiple point
   defects in a rolling element bearing," *Journal of Sound and Vibration*, 1985.

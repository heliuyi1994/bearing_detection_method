# 06 时频分析：STFT 与连续小波变换

> 对应代码：`src/bearing_fault/timefreq.py`
> 测试：`tests/test_timefreq.py`

第 02 章建立的信号模型告诉我们：轴承局部故障的振动特征是**一串瞬态冲击**
激起的结构衰减振荡。第 03、04 章的时域指标与包络谱都先对整段信号做某种
"全局压缩"（统计量或 FFT），回答"有没有故障、故障频率是多少"，却回答不了
"冲击发生在什么时刻、是否严格等间隔"。本章介绍把信号同时沿**时间**与
**频率**两个轴展开的工具——短时傅里叶变换（STFT）与连续小波变换（CWT），
它们给出的时频图能直接呈现冲击的发生时刻与重复规律。

本章默认参数与前几章一致：采样频率 $f_s = 12000$ Hz，转速 1797 rpm
（转频 $f_r = 29.95$ Hz），仿真器共振频率 $f_n = 4000$ Hz、阻尼比
$\zeta = 0.05$、滑差 1%、调制深度 0.6，轴承为 SKF 6205
（BPFO = 107.36 Hz、BPFI = 162.19 Hz 等，见第 01 章）。

## 6.1 为什么需要时频分析：非平稳信号

回顾傅里叶变换的定义：

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j2\pi f t}\, dt
$$

基函数 $e^{-j2\pi f t}$ 是**在时间轴上无限延伸**的等幅正弦。这意味着：

1. **时间信息被积分抹平**。$X(f)$ 的每一点都是对整段历史的加权平均，
   频谱能告诉我们"信号里有 4000 Hz 成分"，却无法说明它出现在哪一秒。
2. **瞬态成分被稀释**。一次冲击的持续时间只有毫秒量级（仿真器中衰减时间
   常数 $\tau = 1/(2\pi\zeta f_n) \approx 0.80$ ms），在几秒长的记录里
   它对全局频谱的贡献被噪声与平稳成分淹没——这正是第 04 章要先做共振
   解调的原因。

严格地说，故障信号是**非平稳**的：其统计特性（某一频带内的瞬时能量）随
时间变化，每过一次冲击就突变一次。对非平稳信号，合理的做法是放弃"一张
频谱"的表述，改用二元函数 $T(\tau, f)$ 描述"$\tau$ 时刻附近、频率 $f$
附近的能量有多少"，这就是**时频分析**。STFT 与 CWT 是其中最经典的两种
线性时频表示。

## 6.2 短时傅里叶变换（STFT）

### 6.2.1 定义与直觉

既然全局频谱丢失时间信息，最朴素的补救是：把信号切成许多短段，对每段各
自做 FFT。写成连续形式，就是用一个以 $\tau$ 为中心的窗函数 $w(t-\tau)$
把信号局部化后再变换：

$$
X(\tau, f) = \int_{-\infty}^{\infty} x(t)\, w(t-\tau)\, e^{-j2\pi f t}\, dt
\tag{6.1}$$

各项的物理意义：

- $w(t-\tau)$：**滑动窗**，只保留 $\tau$ 附近一段信号，把"哪一段"这个
  自由度引入变换；
- $e^{-j2\pi f t}$：常规的傅里叶基，把窗内这一段分解到频率 $f$；
- $(\tau, f)$：时频平面坐标。$|X(\tau, f)|$（或其平方，称为**谱图**
  spectrogram）就是"$\tau$ 时刻、$f$ 频率附近的信号强度"。

式 (6.1) 与全局 FFT 的唯一区别就是窗 $w$：若 $w(t)\equiv 1$，STFT 退化
为普通傅里叶变换，时间分辨率完全丧失。窗正是用"看得准不准"换"知道在
哪儿"的装置。

### 6.2.2 窗函数的作用

直接把信号乘上矩形窗会在段边缘产生突变，变换后出现严重的频谱泄漏
（sinc 形旁瓣）。因此工程上都用边缘平滑过渡到零的窗。本项目固定使用
**Hann 窗**（`timefreq.py:30`，`window="hann"`）：

$$
w[n] = \tfrac{1}{2}\Bigl(1 - \cos\tfrac{2\pi n}{N-1}\Bigr),\qquad n = 0,\dots,N-1
$$

时域加窗对应频域卷积：窗内段的频谱等于真实频谱与窗谱 $W(f)$ 的卷积。
Hann 窗的代价是主瓣变宽——其等效噪声带宽约为 $1.5$ 个频率 bin
（矩形窗为 1 个 bin），即频率分辨率进一步下降约 50%。这是平滑边缘抑制
泄漏必须付的账。

### 6.2.3 时间–频率分辨率与不确定性原理

窗长 $N$（采样点数）同时决定两个方向的分辨率，且互相掣肘：

$$
\Delta t = \frac{N}{f_s}\qquad(\text{时间覆盖：窗内事件无法区分先后})
$$

$$
\Delta f = \frac{f_s}{N}\qquad(\text{频率 bin 宽：段长限制谱线间距})
$$

两式相乘得到一个与 $N$ 无关的常数：

$$
\Delta t \cdot \Delta f = 1 \tag{6.2}$$

**算例**（本项目缺省 `nperseg=256`，$f_s = 12000$ Hz）：

$$
\Delta t = \frac{256}{12000} \approx 21.33\ \text{ms},\qquad
\Delta f = \frac{12000}{256} \approx 46.9\ \text{Hz}
$$

想提高频率分辨率（$N$ 加大），时间分辨率必然同比变差，反之亦然。这不是
实现的缺陷，而是一条基本原理——**不确定性原理**（Gabor–Heisenberg）：
对任意单位能量窗 $g(t)$，定义 RMS 时间宽度与 RMS 频率宽度

$$
\sigma_t^2 = \int (t-\bar t)^2 |g(t)|^2\,dt,\qquad
\sigma_f^2 = \int (f-\bar f)^2 |G(f)|^2\,df,
$$

则必有

$$
\sigma_t \cdot \sigma_f \;\ge\; \frac{1}{4\pi} \tag{6.3}$$

推导梗概：由微分性质 $g'(t) \leftrightarrow j2\pi f\,G(f)$ 与 Parseval
恒等式，$\sigma_f^2 = \frac{1}{4\pi^2}\|g'\|^2$（设 $\bar f = 0$）；再由
Cauchy–Schwarz 不等式，

$$
\sigma_t^2\,\|g'\|^2 \;\ge\; \Bigl|\int (t-\bar t)\,g\,g'\,dt\Bigr|^2
= \Bigl|\tfrac{1}{2}\int g^2\,dt\Bigr|^2 = \tfrac{1}{4},
$$

（分部积分，边界项为零）代入即得式 (6.3)。等号成立当且仅当
$(t-\bar t)g \propto g'$，即 $g$ 为**高斯函数**——高斯窗是时频面元
面积最小的窗。注意式 (6.3) 用 RMS 宽度、式 (6.2) 用工程上的"窗长 ×
bin 宽"，口径不同，数值 $1 \gg 1/(4\pi) \approx 0.0796$ 并不矛盾：
Hann 窗的真实 RMS 单元面积介于两者之间，且任何窗都无法突破下限
$1/(4\pi)$。

### 6.2.4 本项目实现

`stft_spectrogram`（`timefreq.py:21`）是对 `scipy.signal.stft` 的薄封装：

```python
f, t, z = sps.stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
return t, f, np.abs(z)
```

- `nperseg=256`：每段 256 点（21.33 ms），决定式 (6.2) 的分辨率单元；
- `noverlap=nperseg//2=128`：相邻段重叠 50%，时间轴步长
  $128/12000 \approx 10.67$ ms。重叠不改善分辨率，只让亮纹在时间轴上
  更平滑、不遗漏恰好落在段边界的事件；
- 实信号的 STFT 共轭对称，`sps.stft` 只返回单边谱：$129$ 个频率 bin，
  覆盖 $0 \sim f_s/2 = 6000$ Hz；
- 返回幅值 $|X(\tau, f)|$ 而非复数，供直接成像。

## 6.3 连续小波变换（CWT）

### 6.3.1 从固定窗到伸缩窗

STFT 的窗长一旦选定，整个时频平面上分辨率处处相同（均匀网格）。但故障
诊断面对的需求是**不对称**的：

- 低频成分（转频、保持架频率，几十 Hz）周期长，需要长窗才能测准频率；
- 高频冲击（共振带 $\sim$4 kHz，毫秒级）持续极短，需要短窗才能定位时刻。

小波变换的思路是：不用同一个窗，而用**同一个母小波 $\psi$ 的伸缩副本**
——分析低频时把它拉长（长窗），分析高频时把它压短（短窗）。窗的形状
不变，只是尺度在变。

### 6.3.2 定义与能量归一化

$$
W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t)\,
\psi^{*}\!\left(\frac{t-b}{a}\right) dt \tag{6.4}$$

- $a > 0$：**尺度**。$a$ 大则小波被拉宽（分析低频），$a$ 小则被压窄
  （分析高频）；
- $b$：**平移**，即小波中心对准的时刻，对应 STFT 的 $\tau$；
- $\psi^{*}(\cdot)$：母小波的复共轭，$W(a,b)$ 是 $x$ 与该伸缩平移副本
  的**内积（相关）**——副本与信号局部越像，$|W|$ 越大；
- $1/\sqrt{a}$：能量归一化因子。令 $u = (t-b)/a$ 换元可得 $\|\psi_{a,b}\|_2^2 = \frac{1}{a}\int|\psi(u)|^2 \cdot a\,du = \|\psi\|_2^2$，
  即所有尺度的小波能量相同，不同尺度间的幅值可直接比较。

### 6.3.3 复 Morlet 小波

本项目采用复 Morlet 小波（`timefreq.py:34`）：

$$
\psi(t) = \pi^{-1/4}\, e^{j\omega_0 t}\, e^{-t^2/2} \tag{6.5}$$

三项各有分工：

- $\pi^{-1/4}$：**归一化常数**，使 $\|\psi\|_2 = 1$。验证：
  $|\psi|^2 = \pi^{-1/2} e^{-t^2}$，而 $\int e^{-t^2}dt = \sqrt{\pi}$，
  故 $\int|\psi|^2dt = \pi^{-1/2}\cdot\sqrt{\pi} = 1$；
- $e^{j\omega_0 t}$：**复载波**，让小波成为一个中心在角频率 $\omega_0$
  的带通滤波器（高斯的傅里叶变换仍是高斯：
  $\Psi(\omega) \propto e^{-(\omega-\omega_0)^2/2}$）；
- $e^{-t^2/2}$：**高斯包络**，把小波局部化在 $t=0$ 附近。高斯正是使
  不确定性式 (6.3) 取等号的函数，所以 Morlet 小波的时频面元面积达到
  理论最小值 $1/(4\pi)$。

**$\omega_0 = 6$ 的折中**（`morlet_wavelet` 的缺省 `w0=6.0`）：

1. *容许性*。数学上要求小波零均值 $\int\psi\,dt = 0$；严格零均值的
   Morlet 需加修正项，但当 $\omega_0$ 足够大时可直接省略：
   $\int\psi\,dt = \pi^{-1/4}\sqrt{2\pi}\,e^{-\omega_0^2/2}$，
   $\omega_0 = 6$ 时 $e^{-18} \approx 1.5\times10^{-8}$，数值上就是零
   （实测截断离散小波 $|{\rm mean}| \approx 10^{-5}$，远小于峰值
   $\pi^{-1/4} \approx 0.75$，与 `tests/test_timefreq.py` 的断言一致）。
2. *时频分配*。高斯包络的 RMS 宽度为 $1/\sqrt{2}$（$t$ 域）与
   $1/\sqrt{2}$（$\omega$ 域），映射到物理单位（见下节）：

   $$
   \sigma_\tau = \frac{a}{\sqrt{2}\,f_s} = \frac{\omega_0}{2\sqrt{2}\,\pi f},
   \qquad
   \sigma_f = \frac{f}{\omega_0\sqrt{2}} \tag{6.6}$$

   乘积恒为 $1/(4\pi)$，与 $\omega_0$ 无关——$\omega_0$ 不改变面元面积，
   只改变**分配比例**：$\omega_0$ 大则每个振荡周期内包含更多周期
   （频率测得准、时间窗变长），$\omega_0$ 小则相反。$\omega_0 = 6$
   给出相对带宽 $\sigma_f/f = 1/(6\sqrt{2}) \approx 11.8\%$
   （恒 $Q \approx \omega_0\sqrt{2} \approx 8.5$），是文献中定位瞬态
   冲击的常用折中。

### 6.3.4 尺度–频率映射

尺度 $a$ 是伸缩倍数，工程上更想要频率刻度。$\psi((t-b)/a)$ 相对于 $t$
的瞬时角频率是 $\omega_0/a$（每单位 $t$ 转过 $\omega_0/a$ 弧度）。若
$t$ 以**采样点**计，则物理角频率为 $\omega_0 f_s/a$ rad/s，对应

$$
\boxed{\;f = \frac{\omega_0\, f_s}{2\pi\, a}\;}
\qquad\Longleftrightarrow\qquad
a = \frac{\omega_0\, f_s}{2\pi\, f} \tag{6.7}$$

这正是 `cwt_spectrogram` 中的一行（`timefreq.py:64`）：

```python
scale = w0 * fs / (2.0 * np.pi * f)
```

用户传入关注的频率数组 `freqs`，代码逐个换算成尺度并构造小波。频率必须
落在 $(0, f_s/2)$ 内，否则抛出 `ValueError`（`timefreq.py:60`）。

**算例**（$f_s = 12000$ Hz，$\omega_0 = 6$）：

| $f$ (Hz) | $a$（采样点） | $\sigma_\tau$（式 6.6） | $\sigma_f$（式 6.6） | 支撑区 $\pm 4a$ |
|---:|---:|---:|---:|---:|
| 100   | 114.59 | 6.75 ms | 11.8 Hz  | $\pm$38.2 ms |
| 1000  | 11.46  | 0.675 ms | 117.9 Hz | $\pm$3.82 ms |
| 4000  | 2.86   | 0.169 ms | 471.4 Hz | $\pm$0.95 ms |

### 6.3.5 多分辨率：低频长窗、高频短窗

上表体现了 CWT 的核心性质——**恒 $Q$（多分辨率）**：$\sigma_f/f$ 是
常数，频率越低窗越长（频率分辨率高、时间分辨率低），频率越高窗越短。
对比 STFT 在 $nperseg=256$ 下全频段固定 $\Delta t = 21.33$ ms /
$\Delta f = 46.9$ Hz：

- 在 100 Hz 附近，CWT 的 $\sigma_f \approx 11.8$ Hz，比 STFT 的
  46.9 Hz 细 4 倍，代价是 $\sigma_\tau \approx 6.75$ ms；
- 在 4000 Hz 共振带，CWT 的 $\sigma_\tau \approx 0.17$ ms，比 STFT 的
  21.33 ms 细两个数量级——这正是定位故障冲击所需要的；代价是
  $\sigma_f \approx 471$ Hz，但在共振带内我们关心的是"何时"而非
  "多准的频率"，完全可接受。

两种变换都在不确定性原理的预算内，只是**面元形状**不同：STFT 全平面
统一，CWT 按频率自适应。

## 6.4 FFT 快速卷积实现

SciPy 自 1.12 起弃用、1.15 起正式移除 `scipy.signal.cwt`（其依赖的
`signal.morlet` 等一并移除），因此本项目自行实现复 Morlet CWT
（`timefreq.py:41` 的 `cwt_spectrogram`）。

**从定义到卷积。** 把式 (6.4) 离散化（$t, b$ 以采样点计）：

$$
W(a, b) = \frac{1}{\sqrt{a}} \sum_{n} x[n]\,
\psi^{*}\!\left(\frac{n-b}{a}\right) \tag{6.8}$$

注意求和式中 $\psi$ 的自变量是 $n - b$：这正是一个**卷积**。定义翻转
序列 $\tilde\psi[n] = \psi[-n]$，则

$$
W(a, \cdot) = \frac{1}{\sqrt{a}}\; x \star \operatorname{conj}\bigl(\tilde\psi\bigr)
\tag{6.9}$$

即"与翻转共轭小波的卷积 = 与小波的相关"。对应实现
（`timefreq.py:66`）：

```python
out[i] = sps.fftconvolve(x, np.conj(psi[::-1]), mode="same") / np.sqrt(scale)
```

- `psi[::-1]` 完成时间翻转，`np.conj` 取共轭，合起来就是
  $\psi^*(-t)$；
- `sps.fftconvolve` 按卷积定理用 FFT 计算：直接相关每尺度需
  $O(N\cdot L)$（$L$ 为小波长度），FFT 卷积只需 $O(N\log N)$。
  小波最长可达近千点（100 Hz 处 917 点），几十个频点下加速非常明显；
- `mode="same"` 截取与输入等长的中段，使 $W$ 的第 $b$ 列对应原始
  时刻 $b/f_s$；
- 最后除以 $\sqrt{a}$ 补上式 (6.4) 的能量归一化。

**小波的截断。** `morlet_wavelet`（`timefreq.py:34`）把无限长的高斯包络
截断到 $\pm 4a$ 个采样点（`half = max(4, int(4.0*scale))`）：包络在
$|t| = 4a$ 处已衰减到峰值的 $e^{-8} \approx 3.4\times10^{-4}$，截断
误差可忽略；`max(4, …)` 保证高频小尺度时仍有至少 9 个点维持波形。

**正确性验证。** 对 512 点随机信号、$f = 1000$ Hz，逐一按定义式 (6.8)
直接求和，与 `fftconvolve` 路径逐点比较，最大偏差约 $10^{-15}$（双精度
浮点极限）——实现与定义严格一致。`tests/test_timefreq.py` 进一步验证了
正弦脊线落在正确频点、以及冲击时刻的高频能量显著高于无冲击区段。

## 6.5 时频图解读指南

用仿真器生成外圈故障信号（第 02 章模型），观察时频图应看到：

1. **垂向亮纹**。一次冲击近似 $\delta$ 函数，频谱极宽；它激起 $f_n \approx 4000$ Hz 附近的结构共振（衰减常数
   $\tau = 1/(2\pi\zeta f_n) \approx 0.80$ ms）。因此在时频图上，每次
   冲击表现为一条以共振带（约 3–5 kHz）最亮、沿频率方向展开的**垂直
   亮纹**，亮纹在时间上的宽度由衰减常数与变换的时间分辨率共同决定。

2. **亮纹间隔 = 故障周期**。相邻亮纹的时间间距即相邻冲击间隔：
   外圈 $1/\mathrm{BPFO} = 9.31$ ms，内圈 $1/\mathrm{BPFI} = 6.17$ ms，
   滚动体 $1/\mathrm{BSF} = 14.17$ ms。滑差（1%）使间距有微小抖动。
   实测验证：仿真外圈信号 2 s 内含 215 次冲击，平均间隔 9.31 ms，
   与理论值一致；在 4000 Hz 尺度行的 CWT 幅值上，前 8 次冲击全部对应
   $\pm 3.3$ ms 内的亮纹。

3. **亮度的慢起伏 = 调制**。内圈故障幅值随轴转频调制、滚动体故障随
   保持架频率调制（深度 0.6，见第 02 章）：亮纹高度将分别以周期
   $1/f_r = 33.4$ ms 与 $1/\mathrm{FTF} = 83.8$ ms 起伏。外圈故障
   无调制，亮纹高度均匀。

4. **选哪个变换？** 用缺省 STFT（$nperseg=256$）看上述信号：单窗
   21.33 ms 已覆盖 2~3 次外圈冲击（间隔 9.31 ms），**单次冲击不可
   分辨**，只能看到共振带能量整体增强。换 CWT 并把频点放在共振带
   （如第 04 章解调频带 $(2000, 5400)$ Hz 内的对数间隔频点），
   0.17 ms 级的时间分辨率能把每次冲击逐一分开。建议的频点范围须满足
   $0 < f < f_s/2$。

5. **变速工况**下亮纹间距随转速漂移，时间轴上的等间隔判据失效——
   应先在角域分析（第 05 章），时频图此时用于定性观察共振带激发。

时频图给出的"故障周期"假设，最终仍应回到包络谱（第 04 章）与谱峰匹配
评分（第 07 章）做定量确认。

## 6.6 STFT 与 CWT 对比与选用建议

| 维度 | STFT（`stft_spectrogram`） | CWT（`cwt_spectrogram`） |
|---|---|---|
| 时频网格 | 均匀：全平面固定 $\Delta t \times \Delta f$ | 恒 $Q$：低频长窗、高频短窗 |
| 缺省分辨率（$f_s$=12 kHz） | $\Delta t$=21.33 ms，$\Delta f$=46.9 Hz（$N$=256） | 100 Hz 处 6.75 ms / 11.8 Hz；4 kHz 处 0.17 ms / 471 Hz |
| 基函数 | 加窗正弦（窗长固定） | Morlet 小波伸缩平移（窗长随尺度变） |
| 频率轴 | 均匀 bin，$0\sim f_s/2$ 共 129 点 | 用户指定 `freqs`，可任意（如对数间隔、共振带加密） |
| 单次冲击可分辨性 | 差（窗长 > 故障周期） | 好（共振带 $\sigma_\tau \approx 0.17$ ms $\ll$ 9.31 ms） |
| 谐波/边带测频 | 好（低频段等带宽） | 低频段也好，但高频段频率粗 |
| 计算量 | 每帧一次 FFT，便宜 | 每频点一次 FFT 卷积，较贵 |
| 实现 | `scipy.signal.stft` | 自行实现（SciPy ≥1.15 已无 `signal.cwt`） |

**选用建议**：

- 要**定位瞬态冲击、观察共振带激发、估计故障周期** → CWT，频点聚焦在
  共振带（本项目约 $(2000, 5400)$ Hz，覆盖 $f_n = 4000$ Hz）；
- 要**观察谐波与边带结构、做谱图概览、与其他频域结果对照** → STFT，
  并用 `nperseg` 显式权衡分辨率（如 1024 点把 $\Delta f$ 压到
  11.7 Hz，代价是 $\Delta t$ 升至 85 ms）；
- 实践中两者互补：STFT 快速概览，CWT 在共振带细看冲击序列。

## 6.7 小结

- 故障冲击是瞬态非平稳成分，全局 FFT 丢失"何时发生"，需要时频分析；
- STFT 以固定窗滑动，分辨率单元 $\Delta t \cdot \Delta f \approx 1$
  处处相同，受不确定性原理 $\sigma_t\sigma_f \ge 1/(4\pi)$ 约束；
- CWT 用伸缩的复 Morlet 小波，尺度-频率映射
  $f = \omega_0 f_s / (2\pi a)$，天然多分辨率：共振带处时间分辨率达
  0.17 ms，可逐次分辨 9.31 ms 间隔的外圈冲击；
- 本项目基于 FFT 快速卷积自行实现 CWT（SciPy ≥1.15 已移除
  `signal.cwt`），实现与定义式逐点一致（偏差 $\sim 10^{-15}$）；
- 时频图上冲击 = 共振带处的垂向亮纹，亮纹间隔 = 故障周期，亮度起伏
  = 调制；定量判别交给包络谱与第 07 章的诊断流程。

## 参考文献

1. Randall, R. B., & Antoni, J. (2011). Rolling element bearing
   diagnostics—A tutorial. *Mechanical Systems and Signal Processing*,
   25(2), 485–520.
2. McFadden, P. D., & Smith, J. D. (1984). Model for the vibration
   produced by a single point defect in a rolling element bearing.
   *Journal of Sound and Vibration*, 96(1), 69–82.
3. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.
4. Mallat, S. (2009). *A Wavelet Tour of Signal Processing: The Sparse
   Way* (3rd ed.). Academic Press.
5. Cohen, L. (1995). *Time-Frequency Analysis*. Prentice Hall.
6. Gabor, D. (1946). Theory of communication. *Journal of the Institution
   of Electrical Engineers*, 93(26), 429–457.
7. Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet
   analysis. *Bulletin of the American Meteorological Society*, 79(1),
   61–78.
8. SciPy 1.15.0 Release Notes（`scipy.signal.cwt` 等过期 API 移除说明），
   https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html

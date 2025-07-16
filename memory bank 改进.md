### ① Minimal 修改：共享 memory bank 跨图像使用

> 不改网络结构，只改 memory 写入和读取逻辑

- 将图 A 的视觉特征（带有 segmentation）编码后写入 memory；
- 图 B 在推理时读取图 A 的 memory 作为 cross-attention 的 key；
- 图 B 的 feature 做为 query；
- 最终完成 cross-image 的 CoSeg 分割。

**实现建议**：
- 将两张图打包为一组 “伪视频”；
- 图 A 作为 `non_cond_frame_outputs[0]`；
- 图 B 设置为当前帧 `frame_idx = 1`，从图 A 中读取 memory。

---

### ② 使用图 A 的 mask 提取 object pointer token，注入 memory

> 类似 SAM2 中 object pointer 的机制，将其作为 prompt 注入图 B 的 attention 中

- 对图 A 中目标的 mask 区域进行 average pooling 得到 object embedding；
- 加入时间/位置编码后，作为 memory bank 的一部分；
- 图 B 通过 cross-attention 读取这些 object prompt，辅助分割。

**实现建议**：
- 复用 `obj_ptr` 的构造方式；
- 将 pointer 作为 memory token 添加到 `_prepare_memory_conditioned_features()` 中。

---

### ③ 提取语义 token 作为 prompt，用于跨图 attention

> 从图 A 中提取“语义 token”，类似 DETR 中 learnable query 的机制

- 将图 A 中不同 mask 区域映射为语义 token；
- 每个 token 表示一种“要找的对象”；
- 这些 token 作为 memory 中的 prompt；
- 图 B 中图像 patch 与这些 token 做 cross attention 实现语义匹配。

**可扩展方向**：
- 支持多对象；
- token 可通过 transformer encoder 编码提纯。

---

### ④ 直接构造联合 memory encoder，实现对 A+B 图的联合编码

> 将两图输入一起编码，实现深度层级的语义对齐与交互

- 图 A 与图 B 同时输入 encoder；
- 将其 feature concat 后统一做 attention；
- 输出中仅保留图 B 的特征，输入 decoder 完成分割。

**适合用在**：
- 多图共同分割（Multi-CoSeg）；
- 多视角实例搜索；
- 更深层的语义对齐任务。

---


# wqkv
wqkv linear transformer

## method

为方便推导起见，本文假设所使用batch size 为1，不使用多头注意力。

本文所使用的基于softmax全局池化如下所述：

假设对当前shape为[L, C]的序列X得到shape为[L, C]的Q, K, V。则使用权重矩阵w，得到W = wX，其shape为[L, D]。

则池化结果为:
$$
K' = softmax(W^T) K \\
V' = softmax(W^T) V
$$
从而将shape为[L,C]的 K, V压缩成，shape为[D, C]的$K', V'$。

后续使用Q对[K', V']进行attention，也即
$$
C' = softmax(Q^TK')V'
$$

上面是针对全序列的形式。一般来讲，针对无穷长的序列，需要推导其casual的并行训练形式以及递归推理形式。

注意到$K',V'$的计算存在递归形式。将$exp(W^T)$记为$\begin{pmatrix} w_1 & w_2 & ... & w_l\end{pmatrix}$，$K$记为$\begin{pmatrix} k_1 \\ k_2 \\ ... \\ k_l \end{pmatrix}$
$$
K' = \dfrac{\begin{pmatrix} w_1 & w_2 & ... & w_l\end{pmatrix}}{\sum_i^{l}w_i}\begin{pmatrix} k_1 \\ k_2 \\ ... \\ k_l \end{pmatrix} \\
= \dfrac{\begin{pmatrix} w_1 & w_2 & ... & w_l\end{pmatrix} \begin{pmatrix} k_1 \\ k_2 \\ ... \\ k_l \end{pmatrix}}{\sum_i^{l}w_i} \\
= \dfrac{\sum_i^l w_ik_i}{\sum_i^{l}w_i} \\
= \dfrac{\sum_i^{l-1}w_ik_i + w_lk_l}{\sum_i^{l-1}w_i + w_l}
$$
令$M_{l} = \sum_i^{l}w_ik_i, N_{l} = \sum_i^{l}w_i$，则存在如下的递推式：
$$
K_{l} = \dfrac{M_{l-1} + w_lk_l}{N_{l-1} + w_l}\\
K_{l-1} = \dfrac{M_{l-1}}{N_{l-1}}
$$
从上面的递归公式来看，如果想实现并行训练，必须要用到累积和，这也与已有的方法类似。

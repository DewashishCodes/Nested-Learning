# **Nested-Learning**

### *PyTorch implementation of Google’s Nested Learning & HOPE Architecture*

This repository contains a clean, modular, and fully-interpretable implementation of **Nested Learning (NL)** and the **HOPE architecture** introduced by Google DeepMind.
It includes standalone implementations of:

* **CMS (Contextual Multi-Scale Memory)**
* **Nested Optimizers (GDMemory, MomentumMemory, AdamMemory, DMGD, PreconditionedMomentum)**
* **Self-Modifying MLP (rank-1 & rank-k)**
* **HOPE Block (CMS + Self-Modifying MLP + Linear Attention Fast Memory)**
* **Full HOPE architecture assembly**
* Example training on **Tiny Shakespeare**

This repo aims to make the original research **easy to understand** and **easy to build on**.

---

# 🌟 **Features**

✔ Modular PyTorch implementation of **all major components** of Nested Learning
✔ Clean code broken into **separate notebooks**
✔ Implementations match Google’s paper structure
✔ Self-Modifying MLP with **low-rank ΔW updates**
✔ CMS with **parallel multi-timescale memories**
✔ Linear Attention with **fast KV memory updates**
✔ Compare different nested optimizers
✔ Ready for custom tasks and experiments

---

# 📁 **Repository Structure**

```
Nested-Learning/
│
├── HOPE-implementation.ipynb             # Full HOPE block + assembly
├── cms-implementation.ipynb              # CMS multi-level memory
├── nested-optimizer-implementations.ipynb# GD, Momentum, Adam, DMGD, PCM
├── self-modifying-mlp.ipynb              # Rank-1 & Rank-k ΔW weight updates
├── tiny-shakespear.ipynb                 # Example training on Tiny Shakespeare
│
└── NL.pdf                                # Original Google research paper
└── NL-Handwritten-Notes.pdf              # Handwritten notes for mathematical and theoretical reference                
```

Each component is designed to be **independently testable** and can be imported into larger models.

---

# 🧠 **What is Nested Learning?**

Nested Learning introduces a new way for models to:

* learn **multi-timescale memory**
* perform **context-dependent fast learning** inside the forward pass
* update their **own weights on the fly** using low-rank modifications
* combine slow learning (SGD) + fast learning (inner-loop adaptation)

The HOPE architecture is the first fully-scalable implementation of these ideas.

This repo re-creates the core components in a simplified but faithful manner.

---

# 🔩 **Implemented Components**

### **1. Nested Optimizers**

File: `nested-optimizer-implementations.ipynb`

* `GDMemory`
* `MomentumMemory`
* `AdamMemory`
* **DeepMomentumMemory** (DMGD)
* **PreconditionedMomentumMemory**

These treat optimizer state as **differentiable memory**.

---

### **2. CMS: Contextual Multi-Scale Memory**

File: `cms-implementation.ipynb`

A stack of memory levels updated at different speeds:

![Equation](https://math.vercel.app/?bgcolor=auto&from=m_i%28t%29%20%3D%20%5Calpha_i%20m_i%28t-1%29%20%2B%20%281%20-%20%5Calpha_i%29%20f%28x_t%29%20.svg)

Outputs the aggregated multi-scale context.

---

### **3. Self-Modifying MLP**

File: `self-modifying-mlp.ipynb`

The model predicts a low-rank update to its own weights:

![Equation](https://math.vercel.app/?bgcolor=auto&from=%0AW'%20%3D%20W%20%2B%20u%28h%29%2C%20v%28h%29%5ET%0A.svg)

Implemented in:

* Rank-1
* Rank-k (paper-accurate)

---

### **4. Linear Attention Fast Memory**

Part of `HOPE-implementation.ipynb`

A fast KV memory updated with:

![Equation](https://math.vercel.app/?bgcolor=auto&from=%0A%5Ctext%7BKV%7D%20%5Cleftarrow%20%5Ctext%7BKV%7D%20%2B%20K%5E%5Ctop%20V%0A.svg)

Used as a **long-term associative memory**.

---

### **5. Full HOPE Block**

File: `HOPE-implementation.ipynb`

Combines:

* CMS
* Linear Attention
* Self-Modifying MLP
* FFN + LayerNorm
* Memory dictionary (`cms`, `KV`) handling

This is the main unit used to build Nested Learning models.

---

### **6. Example Training: Tiny Shakespeare**

File: `tiny-shakespear.ipynb`

Runs:

* Toy sequence modeling
* Shows how memories evolve over time
* Demonstrates HOPE block in real inference

---

# 🚀 **Getting Started**

### **Install dependencies**

```bash
pip install torch numpy
```

(Optional Jupyter)

```bash
pip install notebook jupyterlab
```

---

# 🧪 **Quick Usage Example**

### Using HOPE block inside a model:

```python
from hope_block import HOPEBlock

block = HOPEBlock(dim=64, cms_levels=3, rank=4)

x = torch.randn(8, 64)
memories = None

out, new_memories = block(x, memories)
```

---

# 📝 **Roadmap**

* [ ] Add full training loop for HOPE-Transformer
* [ ] Add memory visualizations
* [ ] Add benchmark on character-level tasks
* [ ] Add support for multi-head CMS
* [ ] Add GPT-style stacked HOPE layers

---

# 🤝 **Contributing**

Pull requests are welcome.
If you extend the architecture (multi-head CMS, recurrent HOPE, etc.), feel free to submit!

---

# 📜 **License**

MIT License

---

# 🌐 **References**

* **Nested Learning: Scaling Learning with Nested Architectures** (Google, 2024–2025)
* Original Paper: Included as `NL.pdf`

---

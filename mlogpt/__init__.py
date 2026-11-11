# -*- coding: utf-8 -*-
"""mlogpt -- 把一个小 transformer 展开成 Mindustry mlog 蓝图。

模块：
  reference.py  numpy 参考实现（前向 / 采样 / 权重加载）+ 对拍 CLI
  gen.py        蓝图生成器（--build）与闭环运行入口（--run）
  rig.py        一层前向自动分台的核心单元
  rig_chain.py  关键路径更短的树形归约变体
  mschgen.py    蓝图读写（.msch 格式）
  msch.py       蓝图解析
"""

__all__ = ["reference", "gen", "rig", "rig_chain", "mschgen", "msch"]

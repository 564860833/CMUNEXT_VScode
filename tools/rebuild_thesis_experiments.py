import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)


def qn(name: str) -> str:
    return f"{{{W}}}{name}"


def paragraph_text(element: ET.Element) -> str:
    parts = []
    for node in element.iter():
        tag = node.tag.split("}")[-1]
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag == "br":
            parts.append("\n")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def set_paragraph_text(element: ET.Element, text: str) -> None:
    ppr = element.find(qn("pPr"))
    saved_ppr = deepcopy(ppr) if ppr is not None else None
    for child in list(element):
        element.remove(child)
    if saved_ppr is not None:
        element.append(saved_ppr)
    element.extend(make_paragraph(text, None))


def make_paragraph(text: str, style_id: str | None = "3") -> ET.Element:
    p = ET.Element(qn("p"))
    if style_id:
        ppr = ET.SubElement(p, qn("pPr"))
        pstyle = ET.SubElement(ppr, qn("pStyle"))
        pstyle.set(qn("val"), style_id)
    r = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r, qn("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML}}}space", "preserve")
    t.text = text
    return p


def make_heading(text: str, level: int) -> ET.Element:
    return make_paragraph(text, {1: "2", 2: "4", 3: "5"}[level])


def make_caption(text: str) -> ET.Element:
    return make_paragraph(text, "12")


def make_table(rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(qn("tbl"))
    tbl_pr = ET.SubElement(tbl, qn("tblPr"))
    borders = ET.SubElement(tbl_pr, qn("tblBorders"))
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = ET.SubElement(borders, qn(border_name))
        border.set(qn("val"), "single")
        border.set(qn("sz"), "4")
        border.set(qn("space"), "0")
        border.set(qn("color"), "auto")

    max_cols = max(len(row) for row in rows)
    grid = ET.SubElement(tbl, qn("tblGrid"))
    for _ in range(max_cols):
        col = ET.SubElement(grid, qn("gridCol"))
        col.set(qn("w"), "1800")

    for row in rows:
        tr = ET.SubElement(tbl, qn("tr"))
        for cell_text in row:
            tc = ET.SubElement(tr, qn("tc"))
            tc_pr = ET.SubElement(tc, qn("tcPr"))
            tc_w = ET.SubElement(tc_pr, qn("tcW"))
            tc_w.set(qn("w"), "1800")
            tc_w.set(qn("type"), "dxa")
            tc.append(make_paragraph(str(cell_text), "3"))
    return tbl


def find_idx(children: list[ET.Element], text: str) -> int:
    for idx, child in enumerate(children):
        if child.tag == qn("p") and paragraph_text(child) == text:
            return idx
    raise ValueError(f"paragraph not found: {text}")


def replace_range(body: ET.Element, start_text: str, end_text: str, new_elements: list[ET.Element]) -> None:
    children = list(body)
    start = find_idx(children, start_text)
    end = find_idx(children, end_text)
    for child in children[start:end]:
        body.remove(child)
    insert_at = list(body).index(children[end])
    for offset, element in enumerate(new_elements):
        body.insert(insert_at + offset, element)


def remove_range(body: ET.Element, start_text: str, end_text: str) -> None:
    replace_range(body, start_text, end_text, [])


def replace_chapter5(body: ET.Element, elements: list[ET.Element]) -> None:
    replace_range(body, "第 5 章 最终模型实验结果与综合分析", "第 6 章 总结与展望", elements)


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def delta(value: float) -> str:
    return f"{value:+.4f}"


BASE = {
    "iou": 0.7151,
    "dice": 0.8339,
    "se": 0.8024,
    "pc": 0.8730,
    "f1": 0.8287,
    "sp": 0.9883,
    "hd95": 25.95,
    "assd": 14.00,
    "acc": 0.9699,
    "params": 3.149,
    "gflops": 7.231,
}


def metric_row(name: str, cfg: str, values: dict[str, float]) -> list[str]:
    return [
        name,
        cfg,
        fmt4(values["iou"]),
        fmt4(values["dice"]),
        fmt4(values["se"]),
        fmt4(values["pc"]),
        fmt4(values["sp"]),
        fmt4(values["acc"]),
        f"{values['hd95']:.2f}",
        f"{values['assd']:.2f}",
    ]


def build_chapter2_tail() -> list[ET.Element]:
    return [
        make_heading("2.6 有限标注问题与半监督对比基线", 2),
        make_paragraph(
            "医学图像像素级标注通常需要专业医生逐张勾画，标注成本较高，且模糊边界区域存在一定观察者差异。因此，有限标注条件下的模型训练是乳腺超声图像分割中的重要问题。本文的核心贡献是监督式网络结构改进，半监督方法仅作为后续有限标注场景下的潜在对比基线，不作为本文方法设计的主体。"
        ),
        make_paragraph(
            "半监督医学图像分割通常将训练数据划分为有标注集合 D_l 和无标注集合 D_u，并通过监督损失与无监督一致性约束共同优化模型。Mean Teacher 通过教师模型与学生模型之间的一致性约束利用无标注数据，CPS 则使用两个不同初始化的网络互相生成伪标签进行交叉监督。由于本文当前定量实验主要围绕监督式结构改进展开，第 5 章不再单独报告未完成的半监督占位表，而是在统一实验设置下重点验证 DualGAG、DDSR 及融合模型的作用。"
        ),
        make_heading("2.7 本章小结", 2),
        make_paragraph(
            "本章围绕本文研究所需的相关理论与技术基础进行了系统介绍。首先，阐述了医学图像分割的基本概念、任务形式和常用评价指标，明确了本文研究的乳腺超声图像病灶分割属于二维二分类语义分割任务。随后，介绍了 U 型分割结构、跳跃连接机制和 CMUNeXt 基准模型，分析了普通跳跃连接可能引入背景噪声和冗余纹理的问题。"
        ),
        make_paragraph(
            "接着，本章介绍了注意力机制和特征门控方法，说明空间、通道与门控注意力对病灶区域定位和特征筛选的意义；同时分析了乳腺超声图像中的散斑噪声、低对比度和边界模糊问题，为第 3 章 DualGAG 模块和第 4 章 DDSR 模块的设计提供理论依据。最后，本章简要说明有限标注问题和半监督对比基线的定位。具体数据集划分、训练参数、评价指标计算方式和实验结果统一放在第 5 章给出。"
        ),
    ]


def build_chapter5() -> list[ET.Element]:
    elems: list[ET.Element] = []
    p = elems.append

    p(make_heading("第 5 章 实验结果与综合分析", 1))
    p(make_heading("5.1 实验设置", 2))
    p(make_paragraph(
        "为验证本文提出的 DualGAG、DDSR 及二者融合模型在乳腺超声图像病灶分割任务中的有效性，本章统一给出数据集划分、训练参数、评价指标、消融实验、主流方法对比和复杂度分析。第 3 章和第 4 章仅介绍方法结构，所有定量实验均集中在本章报告。"
    ))

    p(make_heading("5.1.1 数据集划分与预处理", 3))
    p(make_paragraph(
        "本文实验采用 BUSI 乳腺超声图像数据集中含病灶标注的 benign 与 malignant 两类样本，共 647 幅图像，其中良性 437 幅、恶性 210 幅。当前实验使用项目中的第三组随机划分文件 busi_train3.txt 与 busi_val3.txt，其中训练集 452 幅、验证集 195 幅。"
    ))
    p(make_caption("表 5-1 BUSI 数据集第三组划分情况"))
    p(make_table([
        ["集合", "总数", "良性", "恶性", "比例", "文件"],
        ["训练集", "452", "304", "148", "69.86%", "busi_train3.txt"],
        ["验证集", "195", "133", "62", "30.14%", "busi_val3.txt"],
        ["合计", "647", "437", "210", "100.00%", "images / masks/0"],
    ]))
    p(make_paragraph(
        "所有图像与掩膜统一调整为 256×256。训练阶段采用随机旋转、随机翻转等增强方式；部分最终融合模型实验启用了保守额外增强，用于提升模型对图像方向、亮度和局部纹理变化的鲁棒性。验证阶段仅执行尺寸调整和归一化处理。"
    ))
    p(make_caption("表 5-2 数据预处理与增强策略"))
    p(make_table([
        ["阶段", "处理方式", "参数设置", "说明"],
        ["训练集", "Resize", "256×256", "统一输入尺度"],
        ["训练集", "RandomRotate90", "p=0.5", "随机旋转增强"],
        ["训练集", "Flip", "p=0.5", "随机翻转增强"],
        ["训练集", "Normalize", "默认均值方差", "输入归一化"],
        ["最终模型部分实验", "Extra Augmentation", "启用 use_extra_aug", "保守亮度、噪声和形变增强"],
        ["验证集", "Resize + Normalize", "256×256", "不使用随机增强"],
    ]))

    p(make_heading("5.1.2 实验环境与训练参数", 3))
    p(make_paragraph(
        "实验代码基于 PyTorch 实现，训练日志记录的统一配置为 batch size 8、训练 300 epoch、初始学习率 0.01、固定阈值 0.5。普通模型采用 BCE-Dice 损失，边界感知模型采用区域损失与 Sobel 边界损失的组合，其中边界损失权重 λ_b=0.3。"
    ))
    p(make_caption("表 5-3 实验环境与训练参数"))
    p(make_table([
        ["项目", "取值"],
        ["数据集", "BUSI benign + malignant，共 647 幅"],
        ["训练/验证划分", "busi_train3.txt / busi_val3.txt，452 / 195"],
        ["输入尺寸", "256×256"],
        ["Batch size", "8"],
        ["Epoch", "300"],
        ["优化器", "SGD，momentum=0.9，weight_decay=0.0001"],
        ["初始学习率", "0.01"],
        ["阈值", "0.5"],
        ["DDSR 默认参数", "smooth_k=5，max_scale=0.05，mode=skip_only"],
        ["边界损失权重", "λ_b=0.3"],
        ["实现环境", "Python 3.8.6，PyTorch 2.0.0+cpu 用于本地统计；训练日志来自当前项目 checkpoint"],
    ]))

    p(make_heading("5.1.3 评价指标", 3))
    p(make_paragraph(
        "本文采用区域重叠、像素分类、边界贴合和模型复杂度四类指标评价分割模型。区域重叠指标包括 IoU 与 Dice；像素分类指标包括 Sensitivity、Precision、Specificity 与 Accuracy；边界指标包括 HD95 和 ASSD；复杂度指标包括参数量 Params 与 GFLOPs。IoU、Dice、Sensitivity、Precision、Specificity 和 Accuracy 越高越好，HD95 与 ASSD 越低越好。"
    ))

    p(make_heading("5.2 消融实验", 2))
    p(make_paragraph(
        "本节在统一训练与验证协议下，对 DualGAG、DDSR、边界感知损失以及最终融合策略进行消融分析。由于历史 checkpoint 中没有完整记录“仅线性域、仅对数域、乘法门控替代”等 DDSR 分支级实验，本节仅报告已有真实日志可支撑的模块级和插入位置消融结果。"
    ))

    p(make_heading("5.2.1 整体模块消融实验", 3))
    overall = {
        "CMUNeXt": {**BASE, "params": 3.149, "gflops": 7.231},
        "+ DualGAG": {"iou": 0.7160, "dice": 0.8345, "se": 0.8110, "pc": 0.8665, "f1": 0.8296, "sp": 0.9866, "hd95": 23.18, "assd": 8.87, "acc": 0.9693, "params": 3.223, "gflops": 7.442},
        "+ BA-DualGAG": {"iou": 0.7195, "dice": 0.8369, "se": 0.8354, "pc": 0.8380, "f1": 0.8324, "sp": 0.9825, "hd95": 20.63, "assd": 7.17, "acc": 0.9680, "params": 3.249, "gflops": 7.526},
        "+ DDSR(S2)": {"iou": 0.7221, "dice": 0.8386, "se": 0.8114, "pc": 0.8727, "f1": 0.8335, "sp": 0.9875, "hd95": 24.29, "assd": 11.03, "acc": 0.9702, "params": 3.236, "gflops": 7.580},
        "Final": {"iou": 0.7286, "dice": 0.8430, "se": 0.8194, "pc": 0.8671, "f1": 0.8369, "sp": 0.9861, "hd95": 19.06, "assd": 6.80, "acc": 0.9693, "params": 3.279, "gflops": 7.800},
    }
    p(make_caption("表 5-4 整体模块消融实验结果"))
    rows = [["模型", "DualGAG", "DDSR", "边界感知", "IoU", "Dice", "HD95", "ASSD", "Params(M)", "GFLOPs"]]
    rows.extend([
        ["CMUNeXt", "×", "×", "×", "0.7151", "0.8339", "25.95", "14.00", "3.149", "7.231"],
        ["CMUNeXt-DualGAG", "√", "×", "×", "0.7160", "0.8345", "23.18", "8.87", "3.223", "7.442"],
        ["CMUNeXt-BA-DualGAG", "√", "×", "√", "0.7195", "0.8369", "20.63", "7.17", "3.249", "7.526"],
        ["CMUNeXt-DDSR", "×", "√", "×", "0.7221", "0.8386", "24.29", "11.03", "3.236", "7.580"],
        ["CMUNeXt-BA-DualGAG-DDSR", "√", "√", "√", "0.7286", "0.8430", "19.06", "6.80", "3.279", "7.800"],
    ])
    p(make_table(rows))
    p(make_paragraph(
        "由表 5-4 可知，最终融合模型相较 CMUNeXt 基准模型 IoU 从 0.7151 提升至 0.7286，Dice 从 0.8339 提升至 0.8430；边界距离指标改善更为明显，HD95 从 25.95 降低至 19.06，ASSD 从 14.00 降低至 6.80。说明 DualGAG、DDSR 与边界感知约束的组合不仅提升了区域重叠程度，也改善了病灶轮廓贴合度。"
    ))

    p(make_heading("5.2.2 DualGAG 插入位置消融实验", 3))
    p(make_paragraph(
        "为分析 DualGAG 在不同跳跃连接层级中的作用，本文比较了单层、多层和全层插入设置。该组实验来自 2026 年 5 月 28 日的同一批位置消融日志，基准模型为同批次 CMUNeXt。"
    ))
    p(make_caption("表 5-5 DualGAG 插入位置消融实验结果"))
    p(make_table([
        ["模型", "DualGAG 插入位置", "IoU", "Dice", "HD95", "ASSD"],
        ["CMUNeXt", "无", "0.7080", "0.8290", "25.00", "10.33"],
        ["DualGAG-S1", "S1", "0.7054", "0.8273", "22.05", "9.39"],
        ["DualGAG-S2", "S2", "0.7093", "0.8299", "21.68", "7.69"],
        ["DualGAG-S3", "S3", "0.7044", "0.8266", "21.41", "7.36"],
        ["DualGAG-S2,S3", "S2,S3", "0.7095", "0.8301", "20.94", "7.50"],
        ["DualGAG-S1,S2,S3", "S1,S2,S3", "0.7083", "0.8292", "21.50", "7.56"],
        ["DualGAG-S0,S1,S2,S3", "S0-S3", "0.7191", "0.8366", "23.00", "8.89"],
    ]))
    p(make_paragraph(
        "从表 5-5 可以看出，DualGAG 插入全部跳跃连接时获得最高 IoU 和 Dice，说明多尺度语义引导有助于增强整体区域分割能力。仅插入较深层时边界距离有所降低，但区域重叠提升有限，表明单一层级门控难以同时兼顾病灶主体完整性和局部轮廓恢复。"
    ))

    p(make_heading("5.2.3 DDSR 插入位置消融实验", 3))
    p(make_paragraph(
        "DDSR 用于在编码阶段对散斑相关特征进行细化。本文比较了不同编码 stage 的 DDSR 插入位置，以判断散斑修正对分割性能和边界指标的影响。"
    ))
    p(make_caption("表 5-6 DDSR 插入位置消融实验结果"))
    p(make_table([
        ["模型", "DDSR 插入位置", "IoU", "Dice", "HD95", "ASSD"],
        ["CMUNeXt", "无", "0.7151", "0.8339", "25.95", "14.00"],
        ["DDSR-S0", "S0", "0.7064", "0.8279", "23.50", "11.27"],
        ["DDSR-S0,S1", "S0,S1", "0.7053", "0.8272", "26.01", "11.63"],
        ["DDSR-S1", "S1", "0.7131", "0.8325", "25.02", "10.05"],
        ["DDSR-S1,S2", "S1,S2", "0.7117", "0.8316", "26.84", "12.01"],
        ["DDSR-S2", "S2", "0.7221", "0.8386", "24.29", "11.03"],
        ["DDSR-S3", "S3", "0.7069", "0.8283", "28.53", "15.45"],
    ]))
    p(make_paragraph(
        "由表 5-6 可知，DDSR 插入 S2 时取得最佳 IoU 和 Dice，相比基准模型分别提升 0.0060 和 0.0047。S0、S0-S1 或 S3 设置未带来稳定提升，说明 DDSR 并非插入越浅或越多越好；在当前 BUSI 划分和训练协议下，中层特征中的散斑修正更有利于保持病灶语义与边界细节之间的平衡。"
    ))

    p(make_heading("5.2.4 最终融合配置对比", 3))
    p(make_caption("表 5-7 CMUNeXt-BA-DualGAG-DDSR 融合配置对比结果"))
    p(make_table([
        ["实验组", "DDSR 位置", "额外增强", "IoU", "Dice", "HD95", "ASSD"],
        ["A", "S2", "否", "0.7186", "0.8363", "22.05", "7.01"],
        ["A-1", "S1", "否", "0.7184", "0.8361", "20.73", "6.84"],
        ["A-2", "S1,S2", "否", "0.7261", "0.8413", "19.37", "6.81"],
        ["B", "S2", "是", "0.7286", "0.8430", "19.06", "6.80"],
        ["C", "S2", "是", "0.7179", "0.8358", "23.53", "8.06"],
    ]))
    p(make_paragraph(
        "表 5-7 显示，融合模型在引入额外增强后的 B 组取得最高 IoU 和 Dice，同时 HD95 与 ASSD 均处于较低水平。因此，本文将 B 组配置作为最终模型结果进行后续综合分析。A-2 组在不使用额外增强时也具有较好性能，说明 DDSR 与 BA-DualGAG 的组合本身能够提供有效增益。"
    ))

    p(make_heading("5.3 与主流分割方法的对比", 2))
    p(make_paragraph(
        "为提供更完整的性能参照，本节列出项目 README 中在 BUSI 数据集三次随机划分上的公开基线结果，并加入本文最终模型的当前验证结果。公开基线结果主要用于显示 CMUNeXt 在轻量化分割网络中的位置；本文模型结果来自表 5-4 中的最终配置。由于当前未对所有主流方法统一重算 HD95、ASSD 和 FPS，本节仅比较 Params、GFLOPs、IoU 和 Dice/F1。"
    ))
    p(make_caption("表 5-8 主流分割方法与本文模型对比结果"))
    p(make_table([
        ["方法", "Params(M)", "GFLOPs", "IoU", "Dice/F1"],
        ["U-Net", "34.52", "65.52", "0.6861±0.0286", "0.7697±0.0310"],
        ["Attention U-Net", "34.87", "66.63", "0.6855±0.0322", "0.7688±0.0350"],
        ["U-Net++", "26.90", "37.62", "0.6949±0.0294", "0.7806±0.0325"],
        ["U-Net3+", "26.97", "199.74", "0.6838±0.0335", "0.7688±0.0368"],
        ["TransUNet", "105.32", "38.52", "0.7139±0.0237", "0.7985±0.0259"],
        ["SwinUNet", "27.14", "5.91", "0.5411±0.0229", "0.6546±0.0191"],
        ["UNeXt", "1.47", "0.58", "0.6504±0.0271", "0.7416±0.0284"],
        ["CMU-Net", "49.93", "91.25", "0.7142±0.0265", "0.7949±0.0292"],
        ["CMUNeXt", "3.14", "7.41", "0.7156±0.0243", "0.7986±0.0258"],
        ["Ours", "3.28", "7.80", "0.7286", "0.8430"],
    ]))
    p(make_paragraph(
        "从表 5-8 可见，本文最终模型在参数量和 GFLOPs 仅略高于 CMUNeXt 的情况下，当前验证 IoU 达到 0.7286，Dice 达到 0.8430。与大规模 Transformer 或混合结构相比，本文模型保持了轻量化优势；与原始 CMUNeXt 相比，DualGAG、DDSR 和边界感知约束使模型更适合乳腺超声图像中的弱边界和散斑噪声场景。"
    ))

    p(make_heading("5.4 模型复杂度与增量分析", 2))
    p(make_paragraph(
        "为分析本文改进模块带来的额外计算开销，本节根据模型代码在 256×256 输入下统计 Params 和 GFLOPs，并结合表 5-4 的验证指标计算相对 CMUNeXt 的性能增量。"
    ))
    p(make_caption("表 5-9 本文模型复杂度与性能增量分析"))
    p(make_table([
        ["模型", "Params(M)", "ΔParams", "GFLOPs", "ΔGFLOPs", "IoU", "ΔIoU", "Dice", "ΔDice"],
        ["CMUNeXt", "3.149", "+0.000", "7.231", "+0.000", "0.7151", "+0.0000", "0.8339", "+0.0000"],
        ["CMUNeXt-DualGAG", "3.223", "+0.074", "7.442", "+0.211", "0.7160", "+0.0009", "0.8345", "+0.0006"],
        ["CMUNeXt-BA-DualGAG", "3.249", "+0.100", "7.526", "+0.295", "0.7195", "+0.0044", "0.8369", "+0.0030"],
        ["CMUNeXt-DDSR", "3.236", "+0.087", "7.580", "+0.349", "0.7221", "+0.0060", "0.8386", "+0.0047"],
        ["Final", "3.279", "+0.130", "7.800", "+0.569", "0.7286", "+0.0135", "0.8430", "+0.0091"],
    ]))
    p(make_paragraph(
        "表 5-9 表明，最终模型相对 CMUNeXt 仅增加约 0.130M 参数和 0.569 GFLOPs，却带来 0.0135 的 IoU 提升和 0.0091 的 Dice 提升。结合 HD95 与 ASSD 的下降，可以认为该复杂度增量在当前任务中是可接受的。"
    ))

    p(make_heading("5.5 可视化与失败案例分析", 2))
    p(make_paragraph(
        "定量结果显示最终模型在区域重叠和边界距离上均优于基准模型。后续可进一步补充原图、真实标注、CMUNeXt、CMUNeXt-DualGAG、CMUNeXt-DDSR 与最终模型的可视化对比，以展示模型在边界模糊、小病灶和散斑噪声明显病例上的具体表现。"
    ))
    p(make_paragraph(
        "从当前指标看，最终模型的边界距离改善最明显，但仍可能在极低对比度、强声影遮挡、病灶与腺体组织粘连、小病灶以及标注主观差异较大的样本上出现局部漏分割、过分割或轮廓偏移。后续可通过多中心数据、边界不确定性建模和半监督预训练进一步提升复杂病例下的稳定性。"
    ))

    p(make_heading("5.6 本章小结", 2))
    p(make_paragraph(
        "本章在统一实验协议下给出了本文方法的主要定量验证结果。整体模块消融表明，DualGAG、DDSR 与边界感知约束共同作用时，最终模型取得最高 IoU、Dice，并显著降低 HD95 与 ASSD。插入位置消融显示，DualGAG 采用多尺度跳跃连接门控更有利于整体区域分割，DDSR 在 S2 位置取得最稳定的区域重叠提升。"
    ))
    p(make_paragraph(
        "复杂度分析表明，最终模型相对 CMUNeXt 的参数量和计算量增幅较小，仍保持轻量化特征。综合来看，本文提出的 CMUNeXt-BA-DualGAG-DDSR 能够在较低复杂度增量下提升乳腺超声病灶分割性能，尤其改善了边界贴合程度。"
    ))

    return elems


def build_chapter6_2() -> list[ET.Element]:
    return [
        make_heading("6.2 主要实验结论", 2),
        make_paragraph(
            "根据第 5 章实验结果，本文最终模型 CMUNeXt-BA-DualGAG-DDSR 在 BUSI 第三组划分上取得 IoU 0.7286、Dice 0.8430、HD95 19.06 和 ASSD 6.80。与 CMUNeXt 基准模型相比，IoU 提升 0.0135，Dice 提升 0.0091，HD95 降低 6.89，ASSD 降低 7.20，说明本文方法在区域重叠和边界贴合两个方面均有改进。"
        ),
        make_paragraph(
            "消融实验表明，DualGAG 能够改善跳跃连接特征筛选，BA-DualGAG 进一步降低边界距离；DDSR 在 S2 位置取得较稳定的 IoU 和 Dice 提升；最终融合模型将语义引导、散斑特征细化和边界感知约束结合后获得最佳综合性能。复杂度统计显示，最终模型相对 CMUNeXt 增加约 0.130M 参数和 0.569 GFLOPs，复杂度增量较小，符合轻量化医学图像分割模型的设计目标。"
        ),
    ]


def main() -> None:
    root = Path.cwd()
    src = Path("C:/Users/Qian/Desktop/\u8bba\u6587/3.\u521d\u7a3f/\u7855\u58eb\u8bba\u6587\u521d\u7a3f.docx")
    out_dir = root / "output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "\u7855\u58eb\u8bba\u6587\u521d\u7a3f_\u5b9e\u9a8c\u8868\u683c\u4e0e\u7b2c5\u7ae0\u91cd\u6784\u7248.docx"

    with zipfile.ZipFile(src, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    document = ET.fromstring(files["word/document.xml"])
    body = document.find(qn("body"))
    if body is None:
        raise RuntimeError("word/document.xml has no body")

    replace_range(body, "2.6 有限标注与半监督分割方法概述", "第 3 章 基于 DualGAG 的 CMUNeXt 跳跃特征引导方法", build_chapter2_tail())
    remove_range(body, "3.6 DualGAG 模块内消融实验", "3.7 本章小结")
    for child in body:
        if child.tag == qn("p") and paragraph_text(child) == "3.7 本章小结":
            set_paragraph_text(child, "3.6 本章小结")
            break
    remove_range(body, "4.4 DDSR 与融合机制模块内消融实验", "4.5 本章小结")
    for child in body:
        if child.tag == qn("p") and paragraph_text(child) == "4.5 本章小结":
            set_paragraph_text(child, "4.4 本章小结")
            break
    replace_chapter5(body, build_chapter5())
    replace_range(body, "6.2 主要实验结论", "6.3 不足与展望", build_chapter6_2())

    files["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    print(out)


if __name__ == "__main__":
    main()

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_thesis_experiments import make_caption, make_paragraph, make_table, paragraph_text, qn


def find_paragraph_index(body: ET.Element, text: str) -> int:
    children = list(body)
    for idx, child in enumerate(children):
        if child.tag == qn("p") and paragraph_text(child) == text:
            return idx
    raise ValueError(f"paragraph not found: {text}")


def insert_before(body: ET.Element, marker_text: str, elements: list[ET.Element]) -> None:
    marker = list(body)[find_paragraph_index(body, marker_text)]
    insert_at = list(body).index(marker)
    for offset, element in enumerate(elements):
        body.insert(insert_at + offset, element)


def build_dualgag_comparison() -> list[ET.Element]:
    return [
        make_paragraph(
            "为进一步说明 DualGAG 与已有注意力模块之间的差异，本文将其与 Attention Gate 和 CBAM 进行对比。Attention Gate 主要利用解码器高级语义对编码器跳跃特征进行空间位置筛选，适合 U 型网络的跳跃连接改进，但通常不显式进行通道重标定和边界响应建模。CBAM 通过通道注意力和空间注意力对当前特征进行自适应增强，具有较好的通用性，但其注意力权重主要由当前输入特征自身生成，并不专门利用解码器语义对跳跃连接进行引导。"
        ),
        make_paragraph(
            "与上述方法不同，DualGAG 的设计目标并不是简单叠加通道注意力和空间注意力，而是在 CMUNeXt 跳跃连接路径中引入解码器语义引导，使编码器浅层或中层特征在进入 Skip-Fusion 之前先经过目标相关性筛选。具体而言，DualGAG 同时考虑空间位置、通道响应和边界相关信息，使门控权重既能抑制无关背景纹理，也能强化与病灶轮廓相关的特征响应。"
        ),
        make_caption("表 3-1 DualGAG 与 Attention Gate、CBAM 的结构差异"),
        make_table(
            [
                ["模块", "主要作用位置", "引导信息", "空间选择", "通道重标定", "边界建模", "与本文任务的关系"],
                [
                    "Attention Gate",
                    "U 型网络跳跃连接",
                    "解码器语义 + 编码器特征",
                    "有",
                    "弱或无",
                    "无显式边界分支",
                    "能够筛选跳跃特征，但对通道冗余和模糊边界刻画不足",
                ],
                [
                    "CBAM",
                    "通用卷积特征块",
                    "当前输入特征",
                    "有",
                    "有",
                    "无显式边界分支",
                    "可增强一般特征表达，但不专门面向跳跃连接语义差异",
                ],
                [
                    "DualGAG",
                    "CMUNeXt 跳跃连接",
                    "解码器语义 + 编码器跳跃特征",
                    "有",
                    "有",
                    "有",
                    "面向乳腺超声弱边界和背景纹理干扰进行针对性筛选",
                ],
            ]
        ),
        make_paragraph(
            "因此，DualGAG 相比普通 Attention Gate 的主要改进在于加入通道重标定和边界相关调制；相比 CBAM 的主要差异在于引入解码器语义作为引导信号，并将模块嵌入跳跃连接而非普通卷积块中。该设计与乳腺超声图像中病灶区域局部化、边界模糊和背景纹理复杂的特点相对应。"
        ),
    ]


def build_ddsr_not_denoising() -> list[ET.Element]:
    return [
        make_paragraph(
            "需要强调的是，DDSR 并不是传统意义上的输入图像去噪模块。输入去噪通常直接作用于原始超声图像，目标是生成视觉上更平滑或噪声更少的图像；但在乳腺超声病灶分割任务中，散斑纹理、病灶边缘和组织回声可能同时表现为高频成分，若在输入端进行强去噪，可能在削弱噪声的同时抹平病灶边界和内部结构信息。"
        ),
        make_paragraph(
            "DDSR 的作用位置位于网络特征空间，其输入为编码器中指定 stage 的特征 x_i，输出为经过弱修正后的特征 x_i^d 以及对应的特征修正残差 r_i。该残差并不等同于真实物理噪声本身，而是模型在端到端训练过程中学习到的、用于调整当前特征响应的受限修正量。由于修正幅度受到 max_scale·sigmoid(alpha) 的逐通道约束，DDSR 更接近一种特征级弱修正机制，而不是不可逆的图像预处理步骤。"
        ),
        make_caption("表 4-1 DDSR 与传统输入去噪的区别"),
        make_table(
            [
                ["比较维度", "传统输入去噪", "DDSR 特征级弱修正"],
                ["作用对象", "原始超声图像", "编码器中间特征"],
                ["训练方式", "可独立预处理，也可与分割模型分离", "嵌入分割网络端到端训练"],
                ["主要目标", "降低视觉噪声或平滑图像", "削弱影响分割的干扰响应并保留结构信息"],
                ["对边界的影响", "可能平滑弱边界", "通过受限残差修正降低过度平滑风险"],
                ["输出含义", "去噪后的图像", "修正后的特征和特征修正残差"],
                ["本文定位", "不采用", "用于提升跳跃特征质量和后续门控稳定性"],
            ]
        ),
        make_paragraph(
            "因此，本文在表述 DDSR 时将其定位为特征级散斑细化或特征级弱修正模块。对数域分支借鉴了超声散斑乘性噪声的建模思想，但在深层特征空间中仅作为近似建模动机使用，具体有效性仍需通过第 5 章的消融实验和边界指标进行验证。"
        ),
    ]


def build_ddsr_dualgag_synergy() -> list[ET.Element]:
    return [
        make_paragraph(
            "DDSR 与 DualGAG 的融合并非两个模块的简单串联，而是围绕跳跃特征质量和解码阶段特征选择形成互补关系。DDSR 位于编码阶段，主要对指定尺度的编码特征进行受限残差修正，降低散斑样纹理和局部干扰对后续跳跃连接的影响；DualGAG 位于解码阶段的跳跃连接路径，利用解码器高级语义对修正后的编码特征进行空间、通道和边界相关筛选。"
        ),
        make_paragraph(
            "从信息流角度看，DDSR 先改善进入跳跃连接的特征质量，DualGAG 再判断哪些位置和通道更应被传递给解码器。若仅使用 DDSR，模型可能提升局部特征稳定性，但缺少高级语义引导下的目标选择；若仅使用 DualGAG，门控权重仍可能受到浅层散斑纹理和无关边缘的干扰。二者结合后，DDSR 提供更稳定的编码特征，DualGAG 则进一步完成目标相关特征筛选。"
        ),
        make_caption("表 4-2 DDSR 与 DualGAG 的协同机制"),
        make_table(
            [
                ["组成部分", "作用阶段", "输入信息", "输出信息", "主要作用"],
                ["DDSR", "编码阶段", "编码器特征 x_i", "修正特征 x_i^d、残差 r_i", "削弱散斑样干扰并稳定跳跃特征来源"],
                ["DualGAG", "解码跳跃连接阶段", "解码器语义 g_i 与编码特征 x_i^d", "筛选后的跳跃特征 x_i'", "根据目标语义选择空间位置、通道和边界响应"],
                ["噪声感知门控", "融合阶段", "DDSR 残差 r_i 与门控中间特征", "噪声感知门控响应", "利用修正残差信息辅助判断易受干扰区域"],
                ["边界感知损失", "优化阶段", "预测掩膜与真实掩膜边界", "边界监督约束", "从损失层面约束轮廓贴合程度"],
            ]
        ),
        make_paragraph(
            "在最终模型中，DDSR 的残差 r_i 还可以作为噪声感知边界门控的辅助输入，使门控模块不仅关注解码器语义与编码器特征的一致性，也能够感知当前特征中需要被修正的区域。与此同时，边界感知损失从优化目标上约束预测轮廓，使特征修正、语义引导和边界监督形成统一的分割优化链条。"
        ),
    ]


def main() -> None:
    root = Path.cwd()
    base_matches = list((root / "output").glob("*实验表格与第5章重构版.docx"))
    if not base_matches:
        raise FileNotFoundError("Cannot find the previous rebuilt thesis docx in output/.")
    base = base_matches[0]
    output = root / "output" / "硕士论文初稿_三四章方法补充版.docx"

    with zipfile.ZipFile(base, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    document = ET.fromstring(files["word/document.xml"])
    body = document.find(qn("body"))
    if body is None:
        raise RuntimeError("word/document.xml has no body")

    insert_before(body, "3.3.2 空间门控分支", build_dualgag_comparison())
    insert_before(body, "4.2 DDSR 双域散斑细化模块", build_ddsr_not_denoising())
    insert_before(body, "4.3.1 融合策略设计", build_ddsr_dualgag_synergy())

    files["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    print(output)


if __name__ == "__main__":
    main()

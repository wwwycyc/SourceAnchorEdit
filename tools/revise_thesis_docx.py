from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    if prefix != "w":
        raise ValueError(f"Unsupported prefix: {prefix}")
    return f"{{{W_NS}}}{local}"


def set_text_element_text(elem: ET.Element, text: str) -> None:
    elem.text = text
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        elem.set(f"{{{XML_NS}}}space", "preserve")


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    ppr = paragraph.find("w:pPr", NS)
    for child in list(paragraph):
        if ppr is not None and child is ppr:
            continue
        paragraph.remove(child)
    run = ET.Element(qn("w:r"))
    text_elem = ET.SubElement(run, qn("w:t"))
    set_text_element_text(text_elem, text)
    paragraph.append(run)


def set_table_cell_text(cell: ET.Element, text: str) -> None:
    tcpr = cell.find("w:tcPr", NS)
    for child in list(cell):
        if tcpr is not None and child is tcpr:
            continue
        cell.remove(child)
    paragraph = ET.Element(qn("w:p"))
    run = ET.SubElement(paragraph, qn("w:r"))
    text_elem = ET.SubElement(run, qn("w:t"))
    set_text_element_text(text_elem, text)
    cell.append(paragraph)


def iter_blocks(body: ET.Element) -> list[tuple[int, str, ET.Element]]:
    blocks: list[tuple[int, str, ET.Element]] = []
    index = 0
    for child in list(body):
        if child.tag == qn("w:p"):
            index += 1
            blocks.append((index, "p", child))
        elif child.tag == qn("w:tbl"):
            index += 1
            blocks.append((index, "tbl", child))
    return blocks


PARAGRAPH_REPLACEMENTS: dict[int, str] = {
    8: "Research and Implementation of Image Editing Methods Based on Diffusion Models",
    56: "针对上述问题，本文提出一种基于源锚定策略控制的扩散图像编辑方法。该方法以源图像的反演 latent 轨迹为结构参考，在每个去噪步中分别计算源分支与目标分支噪声预测，并联合噪声差异、交叉注意力和潜在漂移构造当前步动态编辑证据；在此基础上，将动态编辑证据与语义 ROI 先验融合为有效编辑掩码，用于执行显式的源–目标噪声混合，从而实现局部语义控制。在调度器更新之后，进一步通过后步源锚定将非编辑区域的 latent 拉回源轨迹，从而抑制背景漂移并增强结构保持能力。为了提高方法的可控性，本文将锚定启用时刻与锚定掩码对硬 ROI 的信任强度设计为可调控制量，使方法能够在编辑充分性与背景保真度之间形成连续可调的权衡关系。",
    62: "To address this issue, this thesis proposes a source-anchoring-controlled diffusion image editing method. The method uses the inverted latent trajectory of the source image as a structural reference. At each denoising step, source-branch and target-branch noise predictions are computed, and a dynamic edit evidence map is constructed from noise discrepancy, cross-attention response, and latent drift. This evidence is fused with a semantic ROI prior to form an effective editing mask for explicit source-target noise blending. After each scheduler update, a post-step source anchoring operation pulls non-edited regions back toward the source latent trajectory, thereby suppressing background drift. In addition, both the anchor activation timing and the trust strength toward the hard ROI are treated as controllable factors, enabling a continuous tradeoff between editing adequacy and background preservation.",
    74: "本文围绕“基于扩散模型的语义引导图像编辑”展开，重点研究背景保真问题，具有较强的理论意义和应用价值。从理论上看，该研究有助于深入理解扩散模型在语义编辑中的信息传播机制，以及编辑区域与非编辑区域之间的耦合关系；从应用上看，提高背景保真能够增强编辑结果的稳定性、真实感和实用性，为人像美化、数字媒体、电商运营、人机交互创作和建筑设计等场景提供技术支持。因此，开展该方向研究具有明显的现实意义。",
    86: "（3）实验与结果分析：收集并整理图像编辑领域常用的公开数据集，在数据集上对所提算法进行测试与评估，设置合理的实验参数，对比所提算法与其他图像编辑算法在背景保持、指令遵循等方面的性能指标，并通过定量分析、定性分析和消融实验验证所提方法的有效性。",
    92: "第3章为基于源锚定策略控制的扩散图像编辑方法，是全文的核心章节。首先给出问题的形式化定义和统一符号约定；接着介绍源图像 DDIM 反演与源–目标双分支噪声预测的基础设定；然后依次阐述当前步动态编辑证据（噪声差异、交叉注意力、潜在漂移）的构造方式、有效编辑掩码与 ROI 先验的融合机制，以及源–目标噪声的显式混合方法；在此基础上介绍源锚定机制，包括后步源锚定、锚定掩码的软硬构造、时间调度策略以及锚定启用时刻对编辑–保真度权衡的影响；最后对本章进行小结。",
    93: "第4章为实验环境与系统实现。介绍实验所采用的数据集（PIE-Bench）及其预处理流程，说明实验环境的软件配置（Anaconda 虚拟环境管理、PyTorch 深度学习框架、Hugging Face Diffusers 扩散模型库），并介绍远程服务器连接工具与可视化界面的实现。",
    120: "这里，μθ(xt,t)和Σθ(xt,t)分别表示由神经网络预测的均值和方差。为了降低训练难度，DDPM通常不直接预测均值，而是让网络预测当前样本中所包含的噪声ϵ[2]。设噪声预测网络为ϵθ(xt,t)，则模型训练目标可写为：",
    140: "2.2.2 编码器与解码器",
    194: "本章其余部分组织如下：3.2 节给出问题定义与符号约定；3.3 节介绍源反演与双分支去噪设定；3.4 节定义当前步动态编辑证据；3.5 节描述有效编辑掩码与噪声混合方式；3.6 节给出源锚定机制与调度策略，并在 3.6.4 小节分析锚定策略对编辑充分性与背景保真度之间权衡的影响；3.7 节给出本章小结。",
    240: "其中，wd、wa 和 wc 为加权系数。式（3-11）用于刻画噪声差异、交叉注意力与潜在漂移三者的综合作用；在当前实现中，潜在漂移项的抑制强度主要通过固定权重控制。随后，对原始响应图进行局部平滑，并通过 Sigmoid 变换得到动态编辑证据：",
    250: "动态编辑证据Φt提供当前步的即时响应，而ROI先验Mroi提供全局语义引导。为在两者之间建立平衡，本文在实现中先利用硬 ROI 对动态编辑证据进行支撑约束，再与软 ROI 先验进行插值，得到有效编辑掩码：",
    252: "在去噪早期，较大的融合权重使系统更多依赖 ROI 先验以保持稳定；在去噪后期，较小的融合权重使系统更多依赖经过硬 ROI 约束的当前步动态证据，以提高局部适应性。",
    277: "由式（3-17）和式（3-18）可以看出，锚定掩码Bt直接控制了每一步编辑 latent 与源 latent 之间的混合程度，因此是决定编辑充分性与背景保真度之间权衡关系的核心控制量。较小的hstart使早期锚定更接近软 ROI，整体上更加偏向背景保持；较大的hstart则更早地信任硬 ROI，在编辑区域内部给予目标语义更充分的写入空间，但与此同时也会减弱对边界与背景区域的保守约束。",
    278: "除hstart外，实际实现还设置了锚定启用起点参数source_anchor_start，用于控制显式源锚定从何时开始逐步介入。较小的source_anchor_start意味着更早启用源锚定，更有利于抑制背景漂移；较大的source_anchor_start则允许目标语义在前期更充分展开，但也会提高非编辑区域受到扰动的风险。",
    279: "因此，hstart与source_anchor_start共同构成本文方法的重要控制轴：前者决定锚定掩码在软 ROI 与硬 ROI 之间的空间硬度，后者决定源锚定介入的时间阶段。通过联合调节这两个量，系统可以在“更强背景保持”与“更强编辑充分性”之间形成连续可调的工作点。后续实验在固定source_anchor_start的前提下，重点分析hstart对编辑行为的影响。",
    281: "本章提出了一种基于源锚定策略控制的扩散图像编辑方法。该方法首先通过DDIM反演获得与源图像一致的初始噪声及逐步 latent 轨迹；随后在每个去噪步中分别计算源分支与目标分支噪声预测，并结合噪声差异、交叉注意力和潜在漂移构造当前步动态编辑证据；在语义 ROI 先验的辅助下，动态编辑证据被进一步融合为有效编辑掩码，并用于执行显式的源–目标噪声混合；在调度器步进之后，本文通过源锚定机制将非编辑区域逐步拉回源 latent 轨迹，从而在 latent 层面显式抑制背景漂移。进一步地，本文将锚定启用时刻与锚定掩码对硬 ROI 的信任强度定义为可调控制量，从而建立起清晰的编辑充分性–背景保真度权衡轴。",
    282: "4  实验环境与系统实现",
    293: "第三，PIE-Bench提供了与局部编辑相关的掩码标注信息。这对于本文尤为重要，因为本文不仅关注最终图像是否符合目标文本，还关注编辑是否尽可能局限于目标区域。因此，PIE-Bench中的标注可以支持LPIPS、PSNR和locality ratio等局部相关指标的计算，使实验评估更有针对性。",
    320: "在本文工作中，Diffusers库扮演了扩散模型基础设施的角色，具体包括以下几个方面。（1）模型加载与权重管理：本文通过StableDiffusionPipeline及其相关子组件接口加载Stable Diffusion v1.5[4]的预训练权重，包括VAE（AutoencoderKL）、UNet（UNet2DConditionModel）、文本编码器（CLIPTextModel）和分词器（CLIPTokenizer），全部通过from_pretrained()接口以统一的缓存机制管理。（2）调度器配置：本文使用Diffusers提供的DDIMScheduler和DDIMInverseScheduler分别用于编辑阶段的正向去噪和反演阶段的逆向加噪，两者共享噪声调度参数（alpha序列），确保反演轨迹与编辑轨迹的一致性。（3）DiffEdit基准实现：在外部对比实验（第5章）中，DiffEdit的掩码生成和掩码去噪编辑过程直接基于Diffusers提供的StableDiffusionDiffEditPipeline实现，该Pipeline内置了generate_mask()和掩码条件去噪流程，为基线方法提供了标准参考实现。（4）注意力控制接口：本文通过Diffusers提供的attention_processor机制注册自定义的注意力存储与汇总模块，在UNet前向推断过程中同步捕获各层的交叉注意力图，用于动态编辑掩码中的注意力项At计算。这一机制无需修改UNet源码，仅通过替换默认的注意力处理器即可实现，具有较好的可维护性和兼容性。",
    323: "FinalShell是一款集远程连接、文件管理和服务器监控于一体的可视化运维工具，常用于Windows环境下对Linux服务器进行远程访问与管理。该工具支持SSH等常见远程连接协议，能够为用户提供命令行终端、目录浏览、文件传输以及资源占用监控等功能。与单纯的命令行工具相比，FinalShell具备更友好的图形界面，便于在实验运行、结果整理和文件管理过程中进行统一操作。",
    324: "在本文实验环境中，FinalShell主要用于连接远程GPU服务器，完成代码运行、模型推理、日志查看和结果文件管理等工作。通过FinalShell，用户可以直接在本地设备上访问服务器端的工作目录，并对实验输出结果进行下载、整理和备份，从而提高实验开发与管理的效率。",
    325: "对于频繁操作远程服务器、处理大规模视觉数据的开发者来说，FinalShell是提升生产力的理想工具。其中，SSH连接界面如图4.2所示。",
    328: "下图4.3为SSH连接成功后的FinalShell页面。",
    334: "如图4.4所示，系统主页面由图片预览、运行设置面板、运行状态面板、生成结果区域以及用于图像编辑交互的对话框组成。",
    337: "使用时，用户点击上传图像，输入编辑指令后点击“开始生成”按钮即可启动编辑流程。系统会先处理source prompt与target prompt，并在运行状态面板同步显示当前进度；处理完成后，生成结果面板下方会显示最终编辑结果。图4.5给出了“将猫变成狗”的编辑示例。",
    340: "如图4.6所示，系统先对图像进行理解，并根据用户需求生成更合适的提示词；同时，运行状态会在图4.7所示面板中同步更新，最终得到图4.8所示的编辑结果。",
    349: "本文从编辑充分性与背景保持两个维度综合评估编辑质量。所有指标均在PIE-Bench数据集[21]的标注掩码指导下计算。其中，Edit-CLIP和locality ratio反映编辑区域相关性能；LPIPS与PSNR在非编辑区域上计算，用于衡量背景保持；Structure Distance在全图上基于DINO特征的结构距离计算，用于衡量整体结构稳定性。",
    350: "（1）编辑区域CLIP相似度（Edit-CLIP，↑）。以目标提示词为参考，使用标注掩码仅保留编辑图像中的编辑区域，并计算该区域图像与目标提示词之间的CLIP余弦相似度[18]；最终结果按实现缩放为100×cosine similarity。",
    353: "（4）结构距离（Structure Distance，↓）。在全图上基于DINO特征的自相似矩阵差异计算，反映编辑结果与源图像在深层结构关系上的一致性[20]，用于衡量整体结构保持程度。",
    354: "（5）局部性比率（Locality Ratio，↑）。衡量编辑变化在空间上集中于目标区域的程度，它表示源图像与编辑图像之间的空间变化量中，落在标注编辑区域内的比例（值域[0,1]，越接近1越集中）。",
    357: "5.2.1结果展示与定性分析",
    362: "本文提出的方法本质上构成一族由锚定参数控制的工作点。基于参数扫描结果，hstart=1.0在保持显著背景优势的同时取得了最接近DiffEdit的编辑强度，因此本文在主对比实验中选取hstart=1.0作为本文方法的展示工作点，并记为Ours；而在机制消融中，采用默认工作点hstart=0.35的Framework Base作为分析基线。表5.1给出了Ours与三个代表性基线方法在PIE-Bench上的定量对比。",
    366: "首先，本文方法在背景保持与结构稳定性相关指标上全面优于对比基线。与DiffEdit[6]相比，LPIPS从0.039降至0.033（降幅15%），PSNR从26.299升至27.040（升幅2.8%），structure从0.015降至0.010（降幅33%），locality从0.557升至0.560。与P2P[7]和MasaCtrl[13]相比，本文方法的保真度优势更为显著：P2P的LPIPS高达0.166，structure高达0.071；MasaCtrl虽优于P2P，但仍明显不及本文方法。在保持较强编辑能力的同时，本文方法在保真度上表现最优。",
    367: "其次，在编辑强度维度上，本文方法的Edit-CLIP为22.087，略低于DiffEdit的22.573（差距2.1%），但显著高于MasaCtrl的21.389。这一结果符合本文方法的保真度优先定位：在尽可能逼近最强基线编辑强度的同时，以较小的编辑代价换取显著的背景保持提升。P2P虽取得最高的Edit-CLIP（22.638），但其明显劣化的保真度指标表明该编辑强度是以严重牺牲背景为代价获得的。",
    379: "第二，动态掩码提供了可测量的保真度增益，但并非决定性组件。移除动态掩码后，所有保真度指标均有所劣化（LPIPS升7.7%，structure升26.5%），Edit-CLIP上升0.436。该结果表明，动态掩码的核心功能是在ROI内部施加额外的空间选择性：静态ROI对ROI内所有位置统一放行，而动态掩码通过漂移项Ct和当前步即时信号判断，对ROI内部无需编辑的区域保持约束。移除这一机制后，部分被静态ROI覆盖但实际不需编辑的区域被编辑信号渗透，导致保真度指标下降。动态掩码的作用是增强性的：方法在没有它时仍可运行，但加入该机制后保真度更优。",
    380: "第三，源分支是方法的结构性前提。移除源分支后，Edit-CLIP飙升至24.222（全表最高），但保真度指标全面崩溃：LPIPS升至0.175（劣化约6.7倍），structure升至0.081（劣化约13.5倍），PSNR降至17.261。这些极端数值表明：在缺乏源分支提供结构参考的情况下，目标文本驱动的去噪已不再是编辑，而是退化为无约束的重新生成——输出图像在全局范围发生了与源图像无关的语义和色调变化。因此，源分支不是可选的增强模块，而是界定该方法之为“编辑”而非“生成”的结构性前提。",
    381: "从Locality Ratio指标看，移除源锚定后locality从0.542降至0.502，移除源分支后进一步降至0.484，表明编辑变化在缺乏空间约束时倾向于扩散至全图，不再集中于目标区域。动态掩码移除后locality微升至0.553，与Edit-CLIP的上升一致——静态ROI的硬放行使编辑在目标区域内部更均匀，但也因此丢失了对局部无关区域的空间选择性。",
    384: "锚定起始硬度参数hstart（式3-19）是本文重点分析的控制参数之一，用于调节编辑强度与背景保真度之间的权衡。在本节实验中，其余参数（包括source_anchor_start）保持固定，仅通过调节hstart在[0.35,1.00]范围内的取值，定量分析其对各指标的影响曲线。为提高参数扫描效率，本节在固定的140张测试子集上比较不同hstart工作点及DiffEdit的表现，所有方法均在完全相同的样本集合上评测。表5.3给出了在固定hend=1.0条件下，不同hstart取值对应的性能。",
    387: "从表5.3可以清晰观察到一条单调的编辑–保真度权衡曲线。随着hstart从0.35增大至1.00，Edit-CLIP从21.708上升至22.436（升幅3.4%），Locality Ratio从0.567提升至0.589；与此同时，LPIPS从0.025上升至0.030，PSNR从30.109下降至28.640，structure从0.006上升至0.011。也就是说，随着锚定更早信任硬ROI，编辑区域语义响应逐步增强，而背景与结构保真度相应有所让渡。",
    388: "将表5.3数据绘制为以Edit-CLIP为横轴、structure和LPIPS为纵轴的权衡曲线（图5.2），并在图中标注DiffEdit的对应位置作为参考点。",
    391: "从图表中可得出以下结论。首先，本文方法在所有hstart取值下的保真度指标均优于DiffEdit，包括编辑强度最低的配置（hstart=0.35，structure仅0.006，为DiffEdit的38%）。其次，在编辑强度方面，hstart=1.0时本文方法的Edit-CLIP（22.436）已接近DiffEdit（22.865），差距仅1.9%，且保真度仍全面领先（structure低31%，LPIPS低21%）。再次，参数扫描结果表明，本文框架能够通过单一参数在编辑充分性与背景保持之间形成连续可调的工作曲线，并在多个工作点上保持相对DiffEdit的保真度优势。从极致保真度（hstart=0.35）到平衡编辑（hstart=1.0），用户可根据应用需求自由选择。",
    396: "在方法设计上，本文构建了一条由双分支噪声预测、动态编辑证据、显式噪声混合和后步源锚定组成的完整控制链路。其中，双分支预测提供源结构信息与目标语义信息的显式分离，动态编辑证据用于刻画当前步的局部编辑需求，有效编辑掩码负责将语义先验与即时证据融合为可执行的编辑控制量，而源锚定机制则在每一步latent更新之后对非编辑区域进行空间选择性的回拉。基于此，本文进一步将锚定启用时刻与锚定掩码对硬ROI的信任强度设计为可调控制量，使方法能够在编辑充分性与背景保真度之间形成清晰的可调权衡关系。",
    398: "总体而言，本文完成了以下工作：提出了一种基于源锚定策略控制的扩散图像编辑框架；构建了由动态编辑证据、显式噪声混合和后步源锚定组成的统一编辑控制链路；验证了锚定策略对编辑充分性与背景保真度之间平衡关系的重要作用，并展示了通过锚定参数实现连续可调工作点的可行性。本文的研究说明，在扩散图像编辑任务中，引入显式的源锚定控制能够有效提升背景保持能力，并为局部编辑控制提供更强的可解释性和可调性。",
}


PARAGRAPH_TEXT_DELETIONS = {
    "时光荏苒，岁月如梭。当我的手落在键盘上时我明白大学四年的时光已经到了尾声。",
    "在求学之路上给予我最大的帮助的是我的父母，他们在高考时对我的细心照顾，让我可以将最大的精力投入学习当中。因此我才能考上我理想的大学。家庭的负担和生活的压力从来没有使我分心，父母的努力让我深刻理解了他们对我的帮助是任何事物都不能比较的。他们的默默付出和辛勤工作，只为给我创造一个更好的学习环境，让我能够心无旁骛地追求知识。",
    "在大学的四年时光里我十分庆幸找到了志同道合的朋友们。在同一个环境下生活学习，难免会有矛盾和冲突。然而，我们之间总可以找到方法去化解矛盾和解决冲突。这是我在大学里所收获的东西里最重要的，不单单是在大学的朋友们，还有一直陪伴我的朋友们。",
    "最后我要特别向我的导师徐树振表达我深深的感激之情。从论文的选题到撰写，再到修改和完善，每一步都离不开您的悉心指导和帮助。您的教诲不仅让我在学术上有所收获，更让我学会了如何面对挑战和困难。我衷心祝愿您身体健康，事业蒸蒸日上，继续为学术界培养更多的优秀人才。感谢您给予我的一切，我将永远铭记在心。",
}


TABLE_CELL_REPLACEMENTS: dict[int, dict[tuple[int, int], str]] = {
    181: {
        (0, 2): "（2-17）",
    },
    364: {
        (0, 0): "方法",
        (0, 1): "Edit-CLIP↑",
        (0, 2): "LPIPS↓",
        (0, 3): "PSNR↑",
        (0, 4): "Locality↑",
        (0, 5): "Structure↓",
    },
    376: {
        (0, 0): "配置",
        (0, 1): "Edit-CLIP↑",
        (0, 2): "LPIPS↓",
        (0, 3): "PSNR↑",
        (0, 4): "Locality↑",
        (0, 5): "Structure↓",
    },
    386: {
        (0, 0): "hstart",
        (0, 1): "Edit-CLIP↑",
        (0, 2): "LPIPS↓",
        (0, 3): "PSNR↑",
        (0, 4): "Locality↑",
        (0, 5): "Structure↓",
    },
}


def apply_revisions(docx_path: Path) -> Path:
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    backup_path = docx_path.with_name(f"{docx_path.stem}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}{docx_path.suffix}")
    shutil.copy2(docx_path, backup_path)

    with zipfile.ZipFile(docx_path, "r") as zin:
        file_map = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    root = ET.fromstring(file_map["word/document.xml"])
    body = root.find(".//w:body", NS)
    if body is None:
        raise ValueError("Document body not found")

    blocks = {idx: (kind, elem) for idx, kind, elem in iter_blocks(body)}

    for idx, text in PARAGRAPH_REPLACEMENTS.items():
        kind, elem = blocks[idx]
        if kind != "p":
            raise ValueError(f"Block {idx} is not a paragraph")
        set_paragraph_text(elem, text)

    for paragraph in root.findall(".//w:p", NS):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", NS)]
        text = "".join(texts).strip()
        if text in PARAGRAPH_TEXT_DELETIONS:
            parent = None
            for maybe_parent in root.iter():
                if paragraph in list(maybe_parent):
                    parent = maybe_parent
                    break
            if parent is not None:
                parent.remove(paragraph)

    for idx, replacements in TABLE_CELL_REPLACEMENTS.items():
        kind, table = blocks[idx]
        if kind != "tbl":
            raise ValueError(f"Block {idx} is not a table")
        rows = table.findall("./w:tr", NS)
        for (row_idx, cell_idx), text in replacements.items():
            row = rows[row_idx]
            cells = row.findall("./w:tc", NS)
            set_table_cell_text(cells[cell_idx], text)

    file_map["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in file_map.items():
            zout.writestr(name, data)

    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply unified revisions to the thesis docx.")
    parser.add_argument("docx", type=Path)
    args = parser.parse_args()
    backup = apply_revisions(args.docx.resolve())
    print(f"backup={backup}")
    print(f"updated={args.docx.resolve()}")


if __name__ == "__main__":
    main()

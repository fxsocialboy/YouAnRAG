"""Generate the frozen Stage7 labeled and random evaluation datasets.

The random robustness set is deterministic (seed 20260811).  Run this script
only when initially creating the files or after a documented annotation fix;
do not regenerate/select rows based on model scores.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments"
SEED = 20260811


def infer_disaster_type(query: str, sources=(), *, out_of_domain: bool = False) -> str:
    """Assign and freeze a coarse disaster category during dataset creation."""

    if out_of_domain:
        return "out_of_domain"
    text = query + " " + " ".join(sources)
    if "地震" in text:
        return "earthquake"
    if any(term in text for term in ("滑坡", "泥石流", "地质灾害", "落石", "边坡")):
        return "geological"
    if any(term in text for term in ("山洪", "洪涝", "内涝", "暴雨", "防汛", "积水")):
        return "flood"
    if any(term in text for term in ("气象", "台风", "高温", "大风", "冰雹", "天气")):
        return "meteorological"
    return "comprehensive"


def labeled_rows() -> list[dict]:
    rows: list[dict] = []

    def add(query, query_type, sources=(), chunks=(), facts=(), fallback=False):
        rows.append(
            {
                "query_id": f"final_{len(rows) + 1:03d}",
                "query": query,
                "query_type": query_type,
                "disaster_type": infer_disaster_type(query, sources, out_of_domain=fallback),
                "relevant_source_files": list(sources),
                "relevant_chunk_ids": list(chunks),
                "reference_facts": list(facts),
                "expected_fallback": fallback,
                "metadata": {"dataset": "labeled_quality", "frozen_at": "2026-08-11"},
            }
        )

    add("四川山区群众发现滑坡迹象后如何实现提前避险？", "scenario", ["2022 年四川省地质灾害成功避险典型案例.md"], ["2022 年四川省地质灾害成功避险典型案例.md::1"], ["镇村干部通过巡查发现落石和垮塌迹象", "发现险情后及时报告并加强监测巡查", "垮塌迹象明显时立即组织受威胁群众撤离"])
    add("我国为什么需要专门做山洪灾害防治规划？", "scenario", ["全国山洪灾害防治规划简要报告.md"], ["全国山洪灾害防治规划简要报告.md::0"], ["我国山地丘陵面积大且暴雨频发", "复杂地貌与人类活动导致山洪频繁", "山洪威胁生命和基础设施并制约山区发展"])
    add("国外发生破坏性大地震时我国如何快速收集灾情信息？", "scenario", ["国务院办公厅关于我国对国外发⽣ 破坏性⼤地震作出快速反应问题的通知.md"], ["国务院办公厅关于我国对国外发⽣ 破坏性⼤地震作出快速反应问题的通知.md::1"], ["国家地震部门汇总国外大地震灾情", "驻外使领馆和新华社驻外机构等及时通报", "信息包括伤亡、经济损失和建筑物破坏"])
    add("地震灾区恢复生产工作为什么重要？", "scenario", ["国务院办公厅印发关于地震灾区 恢复⽣产指导意⻅的通知.md"], ["国务院办公厅印发关于地震灾区 恢复⽣产指导意⻅的通知.md::1"], ["尽快恢复生产和生活基本条件", "促进灾区经济社会发展", "服务抗震救灾整体目标"])
    add("国家气象灾害应急预案的编制目的是什么？", "exact_fact", ["国家气象灾害应急预案.md"], ["国家气象灾害应急预案.md::1"], ["建立健全气象灾害应急响应机制", "提高防范和处置能力", "最大限度减轻人员伤亡和财产损失"])
    add("气象灾害应急响应命令一般由谁签发？", "exact_fact", ["国家气象灾害应急预案-2.md"], ["国家气象灾害应急预案-2.md::13"], ["Ⅳ级和Ⅲ级原则上由副局长签发", "Ⅱ级和Ⅰ级原则上由局长签发", "局应急办起草并报局领导签发"])
    add("天津第一次自然灾害综合风险普查的总体情况应该查哪份公报？", "short_ambiguous", ["天津市第一次全国自然灾害综合风险普查公报汇编.md"], ["天津市第一次全国自然灾害综合风险普查公报汇编.md::1"], ["查阅天津市第一次全国自然灾害综合风险普查公报汇编", "总体情况由市普查办和市应急管理局发布", "公报汇总调查、评估区划和数据库建设成果"])
    add("深圳市气象灾害四级响应由哪个机构决定启动？", "exact_fact", ["深圳市气象灾害应急预案.md"], ["深圳市气象灾害应急预案.md::48"], ["由市气象灾害指挥部决定是否启动Ⅳ级响应", "启动前会商评估影响和趋势", "相关办公室联合签发"])
    add("镇坪县为什么要编制地震灾害风险评估报告？", "scenario", ["镇坪县地震灾害风险评估报告.md"], ["镇坪县地震灾害风险评估报告.md::1"], ["预案编制建立在风险评估和资源调查基础上", "辨识分析辖区地震风险", "报告是编制县地震应急预案的基本依据"])
    add("国家地震科学数据中心的数据主要服务哪些防灾减灾场景？", "short_ambiguous", ["国家地震科学数据.md"], ["国家地震科学数据.md::0"], ["服务防震减灾基础业务", "支撑地震科学研究", "通过开放共享服务各行业防灾减灾"])
    add("干线公路地质灾害处置通常会采取哪些临时措施？", "scenario", ["干线公路灾害防治工程典型案例.md"], ["干线公路灾害防治工程典型案例.md::1"], ["裂缝回填夯实和砂浆抹面", "彩条布覆盖裂缝", "设置观测点掌握边坡和桥梁病害"])
    add("郑州720特大暴雨调查报告是谁组织形成的？", "exact_fact", ["河南郑州“7·20”特大暴雨灾害调查报告.md"], ["河南郑州“7·20”特大暴雨灾害调查报告.md::0"], ["报告由国务院灾害调查组形成", "报告时间为2022年1月"])
    add("苏州市自然灾害救助预案的信息管理包括哪些工作？", "exact_fact", ["苏州市自然灾害救助应急预案.md"], ["苏州市自然灾害救助应急预案.md::1"], ["灾情报告", "会商核定", "信息发布"])
    add("IV级 应急响应 命令 副局长 签发", "keyword", ["国家气象灾害应急预案-2.md"], ["国家气象灾害应急预案-2.md::13"], ["Ⅳ级响应原则上由副局长签发", "局应急办负责起草响应命令"])
    add("深圳 气象灾害 Ⅳ级响应 启动 市气象灾害指挥部", "keyword", ["深圳市气象灾害应急预案.md"], ["深圳市气象灾害应急预案.md::48"], ["市气象灾害指挥部决定是否启动Ⅳ级响应", "启动决定建立在会商研判基础上"])
    add("洪涝 突发险情 灾情 报告 防汛抗旱指挥部", "keyword", ["洪涝突发险情灾情报告暂行规定.md"], ["洪涝突发险情灾情报告暂行规定.md::1"], ["各级防汛抗旱指挥部负责掌握和报告", "确定专人负责报告", "重大险情包括严重威胁工程或人员安全"])
    add("十四五 国家综合防灾减灾规划 国减发 2022 1号", "keyword", ["国家综合防灾减灾.md"], ["国家综合防灾减灾.md::1"], ["文件为十四五国家综合防灾减灾规划", "文号国减发〔2022〕1号", "国家减灾委员会于2022年6月19日印发"])
    add("山地城市 内涝 山洪 工程措施 非工程措施", "keyword", ["山地城市内涝与山洪灾害综合防御探讨.md"], ["山地城市内涝与山洪灾害综合防御探讨.md::1"], ["山洪与内涝叠加加重灾害", "防治结合工程和非工程措施"])
    add("自然灾害 情况 统计调查制度 统计法 第七条", "keyword", ["自然灾害情况统计调查制度.md"], ["自然灾害情况统计调查制度.md::1"], ["调查对象必须真实准确完整及时提供资料", "不得提供不真实或不完整资料", "不得迟报或拒报"])
    add("广东省 自然灾害救助 省减灾委员会 组织指挥体系", "keyword", ["广东省自然灾害救助应急预案.md"], ["广东省自然灾害救助应急预案.md::1"], ["包括省减灾委员会", "包括省减灾委员会办公室", "包括专家组和工作组"])
    add("气象灾害响应中，Ⅳ级和Ⅱ级命令的签发人有何不同？", "multi_hop", ["国家气象灾害应急预案-2.md"], ["国家气象灾害应急预案-2.md::13"], ["Ⅳ级原则上由副局长签发", "Ⅱ级原则上由局长签发"])
    add("自然灾害信息从基层掌握上报到核定和对外发布，主要经过哪些环节？", "multi_hop", ["洪涝突发险情灾情报告暂行规定.md", "苏州市自然灾害救助应急预案.md"], ["洪涝突发险情灾情报告暂行规定.md::1", "苏州市自然灾害救助应急预案.md::1"], ["基层指挥机构及时掌握并报告", "对灾情开展会商核定", "核定后开展信息发布"])
    add("为什么山地城市需要把内涝与山洪放在一起治理，措施方向是什么？", "multi_hop", ["山地城市内涝与山洪灾害综合防御探讨.md", "全国山洪灾害防治规划简要报告.md"], ["山地城市内涝与山洪灾害综合防御探讨.md::1", "全国山洪灾害防治规划简要报告.md::0"], ["山洪与内涝叠加会加剧灾害", "山区地貌和暴雨使山洪风险突出", "结合工程和非工程措施"])
    add("如何用量子纠错码提高超导量子计算机的逻辑门保真度？", "out_of_domain", fallback=True)
    add("请给我推荐适合训练大语言模型的GPU集群网络拓扑。", "out_of_domain", fallback=True)
    assert len(rows) == 25
    return rows


def random_rows() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    seen_queries: set[str] = set()

    def push(group: str, query: str, expected_fallback=None):
        if query in seen_queries:
            return False
        seen_queries.add(query)
        rows.append(
            {
                "query_id": f"random_{len(rows) + 1:03d}",
                "query": query,
                "query_type": group,
                "disaster_type": infer_disaster_type(query, out_of_domain=group == "out_of_domain"),
                "relevant_source_files": [],
                "relevant_chunk_ids": [],
                "reference_facts": [],
                "expected_fallback": expected_fallback,
                "metadata": {"dataset": "random_robustness", "seed": SEED},
            }
        )
        return True

    hazards = ["台风", "暴雨", "山洪", "地震", "滑坡", "泥石流", "高温", "洪涝", "大风", "冰雹"]
    places = ["山区小学", "沿河村庄", "高层住宅小区", "地下商场", "城市地铁站", "养老院", "高速公路服务区", "工业园区", "旅游景区", "乡镇卫生院"]
    constraints = ["夜间通信中断", "同时有老人和儿童需要转移", "道路部分积水", "现场人员缺少专业设备", "预警时间只剩半小时", "连续降雨已经两天", "附近存在次生灾害风险", "人员情绪比较恐慌"]
    templates = ["{place}遇到{hazard}预警时，负责人第一步应该做什么？", "{hazard}发生后，{place}怎样组织人员安全转移？", "如果{place}面临{hazard}，而且{constraint}，应急处置顺序怎么安排？", "请给出{place}应对{hazard}的简要行动清单。"]
    choices = [(h, p, c, t) for h in hazards for p in places for c in constraints for t in templates]
    rng.shuffle(choices)
    scenario_count = 0
    for hazard, place, constraint, template in choices:
        scenario_count += int(push("scenario", template.format(hazard=hazard, place=place, constraint=constraint)))
        if scenario_count == 40:
            break

    levels, aspects = ["Ⅰ级", "Ⅱ级", "Ⅲ级", "Ⅳ级"], ["启动条件", "签发主体", "响应措施", "信息报告要求", "解除条件"]
    regions = ["国家气象部门", "深圳市", "广东省", "苏州市", "天津市"]
    templates = ["{region} {level} 应急响应 {aspect}", "请问{region}的{level}响应由谁决定，相关{aspect}是什么？", "查一下{level}响应的{aspect}和责任部门。"]
    choices = [(level, aspect, region, template) for level in levels for aspect in aspects for region in regions for template in templates]
    rng.shuffle(choices)
    keyword_count = 0
    for level, aspect, region, template in choices:
        keyword_count += int(push("keyword", template.format(level=level, aspect=aspect, region=region)))
        if keyword_count == 25:
            break

    noisy = ["台风黄预警学校咋整", "山洪来了往哪跑啊", "地振后高楼咋撤", "暴雨地库进水怎么办", "滑坡有点征兆要不要先撤", "四级响应到底谁开", "灾情咋上报不容易漏", "路边山体掉石头咋处理", "天气灾害预案是干啥的", "郑州720报告谁写的", "灾后复产为啥这么急", "风险普查到底查了啥", "山洪和内涝能一起治不", "统计灾情能晚点报吗", "国外大地震国内咋知道情况", "学校遇大风还上课吗", "泥石流把路堵了先干啥", "老人院暴雨咋转移", "地下商场停电又积水咋办", "预警升级了谁来签字"]
    for query in noisy:
        push("short_ambiguous", query)

    pairs = [("国家气象部门", "深圳市"), ("广东省", "苏州市"), ("天津市", "深圳市"), ("国家层面", "地方指挥部"), ("山区村庄", "城市地下空间")]
    for left, right in pairs:
        push("multi_hop", f"比较{left}和{right}在应急响应启动主体与信息上报方面的差异。")
    for hazard in ["地震", "山洪", "台风", "洪涝", "泥石流"]:
        push("multi_hop", f"{hazard}发生后，如何同时安排人员转移、灾情报告和基础设施检查？")
    for place in ["山区学校", "沿河村庄", "旅游景区", "公路沿线社区", "乡镇卫生院"]:
        push("multi_hop", f"如果{place}先遭遇暴雨又出现滑坡迹象，应如何组合两类预案措施？")

    out_of_domain = ["怎么给Transformer模型设计稀疏注意力算子？", "推荐一套4K视频剪辑电脑配置。", "糖尿病患者如何调整胰岛素剂量？", "写一个量化交易的高频套利策略。", "量子计算中的表面码如何纠错？", "如何制作正宗法式可颂？", "帮我分析今年新能源汽车股票走势。", "Java虚拟机的垃圾回收器怎么选？", "怎样训练一个围棋强化学习模型？", "请解释黑洞信息悖论。", "设计一个跨数据中心分布式数据库。", "如何申请美国计算机博士项目？", "给我安排一周增肌饮食计划。", "摄影时怎样控制景深和快门？", "如何修复损坏的NTFS分区？", "写一首关于春天的现代诗。", "怎样优化CUDA矩阵乘法内核？", "推荐几部适合周末看的科幻电影。", "如何设计电商平台优惠券系统？", "解释一下蛋白质折叠预测原理。"]
    for query in out_of_domain:
        push("out_of_domain", query, True)
    assert len(rows) == 120
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled, random_set = labeled_rows(), random_rows()
    write_jsonl(OUT_DIR / "eval_queries_final_labeled.jsonl", labeled)
    write_jsonl(OUT_DIR / "eval_queries_final_random.jsonl", random_set)
    print(json.dumps({"labeled": len(labeled), "random": len(random_set), "seed": SEED}, ensure_ascii=False))


if __name__ == "__main__":
    main()

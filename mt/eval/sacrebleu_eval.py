"""
使用SacreBLEU进行统一评估
"""
from typing import List, Dict, Optional
import sacrebleu
from pathlib import Path


def evaluate_sacrebleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "zh",
    lowercase: bool = False
) -> Dict[str, float]:
    """
    使用SacreBLEU评估翻译质量
    
    Args:
        hypotheses: 系统输出（假设翻译）
        references: 参考翻译
        tokenize: tokenization方法 ("zh", "13a", "none"等)
        lowercase: 是否转换为小写
    
    Returns:
        包含BLEU、chrF、TER等指标的字典
    """
    # 确保references是列表的列表（SacreBLEU要求）
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    
    # 计算BLEU
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        references,
        tokenize=tokenize,
        lowercase=lowercase
    )
    
    # 计算chrF
    chrf = sacrebleu.corpus_chrf(
        hypotheses,
        references,
        tokenize=tokenize,
        lowercase=lowercase
    )
    
    # 计算TER（如果可用）
    try:
        ter = sacrebleu.corpus_ter(
            hypotheses,
            references,
            tokenize=tokenize,
            lowercase=lowercase
        )
        ter_score = ter.score
    except:
        ter_score = None
    
    return {
        "BLEU": bleu.score,
        "BLEU_brevity_penalty": bleu.bp,
        "BLEU_ratio": bleu.ratio,
        "chrF": chrf.score,
        "TER": ter_score,
    }


def evaluate_from_files(
    hyp_file: str,
    ref_file: str,
    tokenize: str = "zh",
    lowercase: bool = False
) -> Dict[str, float]:
    """
    从文件读取并评估
    
    Args:
        hyp_file: 假设翻译文件路径（每行一个翻译）
        ref_file: 参考翻译文件路径（每行一个翻译）
        tokenize: tokenization方法
        lowercase: 是否转换为小写
    
    Returns:
        评估指标字典
    """
    # 读取文件
    with open(hyp_file, 'r', encoding='utf-8') as f:
        hypotheses = [line.strip() for line in f]
    
    with open(ref_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    return evaluate_sacrebleu(hypotheses, references, tokenize, lowercase)


def batch_evaluate_runs(runs_dir: str, ref_file: str, output_file: Optional[str] = None):
    """
    批量评估runs目录下的所有输出
    
    Args:
        runs_dir: runs目录路径
        ref_file: 参考翻译文件路径
        output_file: 输出汇总文件路径（可选）
    """
    runs_path = Path(runs_dir)
    results = []
    
    # 查找所有hyp文件
    for hyp_file in runs_path.rglob("*.hyp.txt"):
        run_name = hyp_file.parent.name
        metrics = evaluate_from_files(str(hyp_file), ref_file)
        results.append({
            "run": run_name,
            "file": str(hyp_file),
            **metrics
        })
    
    # 打印结果
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    for result in results:
        print(f"\n{result['run']}:")
        print(f"  BLEU: {result['BLEU']:.2f}")
        print(f"  chrF: {result['chrF']:.2f}")
        if result['TER'] is not None:
            print(f"  TER: {result['TER']:.2f}")
    
    # 保存到文件
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    return results


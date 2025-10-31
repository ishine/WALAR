import json
import os
from typing import Any, Dict, List, Optional, Tuple
from utils import training_langs, training_langs2, llamax_langs, high_langs, flores_langs, qwen_langs, xcomet_support_langs


def _find_key_recursively(obj: Any, target_key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = _find_key_recursively(value, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursively(item, target_key)
            if found is not None:
                return found
    return None


def read_spbleu_from_json(file_path: str, key="spBLEU") -> Optional[float]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        value = _find_key_recursively(data, key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    except Exception:
        return None


def iter_result_files(dir_path: str) -> List[Tuple[str, str, str]]:
    results: List[Tuple[str, str, str]] = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            if not name.startswith('result_') or not name.endswith('.json'):
                continue
            try:
                pair = name[len('result_'):-len('.json')]
                src, tgt = pair.split('-', 1)
                results.append((src, tgt, os.path.join(root, name)))
            except Exception:
                continue
    return results

if __name__ == "__main__":
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/global_step550_hf/flores"
    dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128_global_step1050_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128_global_step950_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule_no_pl-LlamaX3-8B-schedule_no_pl-1M-bsz128_global_step50_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-8b/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/bsz1536_schedule-llamax3-8B-schedule_mix2-bsz128_global_step200_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/aya-expanse-8b/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Tower-Plus-9B/flores"
    # dir_path = "/mnt/gemini/data1/yatish/BenchMAX/tasks/translation/output/LLaMAX3-8B-Alpaca/flores/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule_reward-LlamaX3-8B-schedule_no_pl-1M-bsz128_global_step550_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/Llama-3.2-3B-Instruct/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/schedule-LlamaX3-8B-schedule-1M-bsz128_global_step550_hf/flores"
    # dir_path = "/mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation/output/final_Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128/flores"
    # Leave these as None to use all available pairs discovered from filenames
    # Otherwise, set to lists like ["en", "ar"] using codes as in the filenames
    src_langs_i_care: Optional[List[str]] = ["en"]
    tgt_langs_i_care: Optional[List[str]] = ["ar", "az", "be", "bn", "ceb", "cs", "cy", "el", "fa", "gu", "hi", "hy", "is", "jv", "lb", "lt", "lv", "mk", "mr", "ne", "pa", "sk", "sl", "tl", "tr", "uk", "uz", "uz"]
    # tgt_langs_i_care: Optional[List[str]] = ["zh", "kn","is","lb","be","cs","mk","sk","sl","uk","bn","gu","hi","mr","ne","pa","ur","hy","el","lv","lt","fa","cy","ceb","tl","jv","ar","az","tr","uz"]  # e.g., ["ar", "az", "uk"]
    # tgt_langs_i_care: Optional[List[str]] = ["ar", "bn", "cs", "de", "es", "fr", "hu", "ja", "ko", "ru", "sr", "sw", "te", "th", "vi", "zh"]
    # tgt_langs_i_care: Optional[List[str]] = ['af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu']
    # tgt_langs_i_care: Optional[List[str]] = ["is", "lb", "be", "cs", "mk", "sk", "sl", "uk", "bn", "gu", "hi", "mr", "ne", "pa", "ur", "hy", "el", "lv", "lt", "fa", "cy", "ceb", "tl", "jv", "ar", "az", "tr", "uz"]

    # src_langs_i_care: Optional[List[str]] = ['af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu']
    # src_langs_i_care: Optional[List[str]] = ["ar", "bn", "cs", "de", "es", "fr", "hu", "ja", "ko", "ru", "sr", "sw", "te", "th", "vi", "zh"]
    # tgt_langs_i_care: Optional[List[str]] = ["zh"]
    results_by_pair: Dict[str, Optional[float]] = {}
    key = "metricx_score"
    # key="xcomet_score"
    # key = "spBLEU"
    print(dir_path)
    print(f"Direction\t{key}")
    # import code; code.interact(local=locals())
    for src, tgt, path in iter_result_files(dir_path):
        if src_langs_i_care is not None and src not in src_langs_i_care:
            continue
        if tgt_langs_i_care is not None and tgt not in tgt_langs_i_care:
            continue
        if tgt not in xcomet_support_langs:
            continue
        score = read_spbleu_from_json(path, key)
        pair_key = f"{src}-{tgt}"
        results_by_pair[pair_key] = score
        print(f"{pair_key}\t{score if score is not None else ''}")

    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    score_list = []
    valid_scores = [s for s in results_by_pair.values() if s is not None]
    if not results_by_pair:
        print("No results found.")
    elif not valid_scores:
        print("No valid spBLEU scores found.")
    else:
        avg_bleu = sum(valid_scores) / len(valid_scores)
        print(f"Average spBLEU: {avg_bleu:.4f}")
        print(f"Total language pairs: {len(valid_scores)}")

        low_perf_pairs = sorted([p for p, s in results_by_pair.items() if s is not None and 1 < s < 20])
        if low_perf_pairs:
            print(f"Low performance language pairs: {low_perf_pairs}")

        print("\nIndividual scores:")
        for pair in sorted(results_by_pair.keys()):
            score = results_by_pair[pair]
            if score is not None:
                print(f"{pair}: {score}")
                score_list.append(score)
    print(f"\nFinal Average {key}: {sum(score_list) / len(score_list):.4f}")

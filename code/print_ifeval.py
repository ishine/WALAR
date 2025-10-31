import json


if __name__ == "__main__":
    data_name = "gpqa"
    file_path = f"/mnt/gemini/data1/yifengliu/BenchMAX/results/{data_name}/__mnt__gemini__data1__yifengliu__checkpoints__final__Final-Qwen3-4B-post_final_mix-320k-1M-bsz128/results_2025-10-28T17-07-00.091462.json"
    # file_path = "/mnt/gemini/data1/yifengliu/BenchMAX/results/__mnt__gemini__data1__yifengliu__model__LLaMAX3-8B-Alpaca/results_2025-10-26T21-22-02.894139.json"
    if data_name == "ifeval":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lang_list = ["ar", "bn", "cs", "de", "en", "es", "fr", "hu", "ja", "ko", "ru", "sr", "sw", "te", "th", "vi", "zh"]
        results = data['results']
        all_results = []
        for lang in lang_list:
            key = f"xifeval_{lang}"
            result = results.get(key, None)
            inst_level_loose_acc = result['inst_level_loose_acc,none']
            inst_level_strict_acc = result['inst_level_strict_acc,none']
            prompt_level_loose_acc = result['prompt_level_loose_acc,none']
            prompt_level_strict_acc = result['prompt_level_strict_acc,none']
            acc = (inst_level_loose_acc+inst_level_strict_acc+prompt_level_loose_acc+prompt_level_strict_acc)/4
            all_results.append(acc)
            print(f"{lang}: {acc:.4f}")
        average_acc = sum(all_results)/len(all_results)
        print(f"Average Acc: {average_acc:.4f}")
    elif data_name == "mgsm":
        with open(file_path, 'r') as f:
            data = json.load(f)
        lang_list = ["ar", "bn", "cs", "de", "en", "es", "fr", "hu", "ja", "ko", "ru", "sr", "sw", "te", "th", "vi", "zh"]
        results = data['results']
        all_results = []
        for lang in lang_list:
            key = f"xmgsm_native_cot_{lang}"
            result = results.get(key, None)
            if result is None:
                continue
            acc = result['exact_match,flexible-extract']
            all_results.append(acc)
            print(f"{lang}: {acc:.4f}")
        average_acc = sum(all_results)/len(all_results)
        print(f"Average Acc: {average_acc:.4f}")
        # import code; code.interact(local=locals())
    elif data_name == "gpqa":
        with open(file_path, 'r') as f:
            data = json.load(f)
        lang_list = ["ar", "bn", "cs", "de", "en", "es", "fr", "hu", "ja", "ko", "ru", "sr", "sw", "te", "th", "vi", "zh"]
        results = data['results']
        all_results = []
        for lang in lang_list:
            key = f"xgpqa_main_native_cot_zeroshot_{lang}"
            result = results.get(key, None)
            if result is None:
                continue
            acc = result['exact_match,flexible-extract']
            all_results.append(acc)
            print(f"{lang}: {acc:.4f}")
        average_acc = sum(all_results)/len(all_results)
        print(f"Average Acc: {average_acc:.4f}")
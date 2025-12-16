export CUDA_VISIBLE_DEVICES=2
declare -A model_path
num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

model_path["Qwen"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B"
model_path["Qwen3X"]="/mnt/gemini/data1/yifengliu/model/Qwen3-XPlus-8B"
model_path["llama"]="/mnt/gemini/data1/yifengliu/model/Llama-3.2-3B-Instruct"
model_path["llamax"]="/mnt/gemini/data1/yifengliu/model/LLaMAX3-8B-Alpaca"
model_path["aya8b"]="/mnt/gemini/data1/yifengliu/model/aya-expanse-8b"
model_path["aya32b"]="/mnt/gemini/data1/yifengliu/model/aya-expanse-32b"
model_path["tower"]="/mnt/gemini/data1/yifengliu/model/Tower-Plus-9B"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/bsz1536_schedule-llamax3-8B-schedule_mix2-bsz128/global_step200_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule1024-LlamaX3-8B-schedule_mix10k-1M-bsz128/global_step1800_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/pure_qe-LlamaX3-8B-schedule_mix10k-1M-bsz128/global_step200_hf"
model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/qe+lang_detect-LlamaX3-8B-schedule_mix10k-1M-bsz128/global_step600_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule-LlamaX3-8B-schedule-1M-bsz128/global_step1000_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128/global_step150_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final-Qwen3-4B-post_final_mix-320k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128/global_step1400_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Final2-mix-LlamaX3-8B-final_llamax_mix-100k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/final/Continue-Final-Llama3.2-3B-post_final_mix-160k-1M-bsz128"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_mix-LlamaX3-8B-schedule_mix-1M-bsz128/global_step250_hf"
# model_path["checkpoint"]="/mnt/gemini/data1/yifengliu/checkpoints/schedule_no_pl-LlamaX3-8B-schedule_no_pl-1M-bsz128/global_step50_hf"
model_path["Qwen-base"]="/mnt/gemini/data1/yifengliu/model/Qwen3-4B-Base"

model_name="checkpoint"
final_path=${model_path[$model_name]}

# src_lang_list=(
#     # "en"
#     "en","sw","zh","ar","tr","hi","ru"
#     # "ar"
#     # "en"
# )
# tgt_lang_list=(
#     # "ar","bn","cs","de","es","fr","hu","ja","ko","ru","sr","sw","te","th","vi","zh","en"
#     # "hu","vi","es","cs","fr","de","ru","bn","sr","ko","ja","th","sw","zh","te","en"
#     # "cs" "de" "fi"
#     # "af","da","nl","de","no","sv","ca","fr","gl","pt","ro","es","bg","ru","it","id","ms","zh","ja","vi"
#     # "zh","kn","is","lb","be","cs","mk","sk","sl","uk","bn","gu","hi","mr","ne","pa","ur","hy","el","lv","lt","fa","cy","ceb","tl","jv","ar","az","tr","uz"
#     'af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu'
# )

src_lang_list=(
    "en"
    # 'af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu'
    # "en","ru","sw","zh"
    # "ja","az","ne"
    # "ar","tr","hi"
    # 'af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu'
    )
# tgt_lang_list=("en","ar","hi","tr","ja")
tgt_lang_list=(
    'af','am','ar','hy','as','ast','az','be','bn','bs','bg','my','ca','ceb','zh','zho_trad','hr','cs','da','nl','en','et','tl','fi','fr','ff','gl','lg','ka','de','el','gu','ha','he','hi','hu','is','ig','id','ga','it','ja','jv','kea','kam','kn','kk','km','ko','ky','lo','lv','ln','lt','luo','lb','mk','ms','ml','mt','mi','mr','mn','ne','ns','no','ny','oc','or','om','ps','fa','pl','pt','pa','ro','ru','sr','sn','sd','sk','sl','so','ku','es','sw','sv','tg','ta','te','th','tr','uk','umb','ur','uz','vi','cy','wo','xh','yo','zu'
    # "ja","az","ne"
    # "en","ru","sw","zh","ar","tr","hi"
    
    # "is","lb","be","cs","mk","sk","sl","uk","bn","gu","hi","mr","ne","pa","ur","hy","el","lv","lt","fa","cy","ceb","tl","jv","ar","az","tr","uz"
    )

cd /mnt/gemini/data1/yifengliu/BenchMAX/tasks/translation
# src_lang_list=(en)
# tgt_lang_list=(is)
# python generate_translation.py -s ${src_lang_list} -t ${tgt_lang_list} --task-name flores --model-name ${final_path} --infer-backend vllm --max-tokens 4096 --output-parser r1_distill
python generate_translation.py -s ${src_lang_list} -t ${tgt_lang_list} --task-name flores --model-name ${final_path} --infer-backend vllm --max-tokens 512 --tensor-parallel-size ${num_gpus}


python evaluate_translation.py -s ${src_lang_list} -t ${tgt_lang_list} --task-name flores --model-name ${final_path} --metrics spBLEU
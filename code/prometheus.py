# Absolute Grading: Outputs score of 1 to 5
import os
import argparse
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE, ABSOLUTE_PROMPT_WO_REF
from utils import preprocess_dataset, write_to_file
  
lang_dict = {
  "eng": "English",
  "zh": "Chinese",
  "assamese": "Assamese",
  "punjabi": "Punjabi",
  "kannada": "Kannada",
  "maithili": "Maithili",
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate a model using Prometheus Eval")
  parser.add_argument("--model_path", type=str, required=True, help="Path to the model to be evaluated")
  parser.add_argument("--input_file", type=str, required=True, help="Path to the input file containing evaluation data")
  parser.add_argument("--turns", type=int, default=1, help="Number of turns for evaluation (default: 1)")
  parser.add_argument("--src", type=str, default="en", help="Source language code (default: 'en')")
  parser.add_argument("--tgt", type=str, default="fr", help="Target language code (default: 'fr')")
  parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output files")
  args = parser.parse_args()
  model = VLLM(model=args.model_path)
  judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF)

  
  ds, name = preprocess_dataset(args.input_file)
  # ds = [
  #   {"source": "Carnegie Mellon University is famous for its computer science and technology.",
  #    "hypothesis": "Carnegie Mellon University is renowned for its computer science and technology programs."
  #   # "hypothesis": "卡耐基梅隆大学以其计算机科学和技术而闻名。",
  #   },
  # ]
  ds = [
    {
      "source": "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added.",
      "hypothesis": "现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。\n\n中文翻译如下：\n\n现在我们有四个月大的老鼠，这些老鼠曾经是糖尿病患者。"
    }
  ]
  # ds = ds[:10]
  instructions, sources, responses = [], [], []
  for data in ds:
    source = data['source']
    response = data['hypothesis']
    instruction = f"Translate the following text from {lang_dict[args.src]} to {lang_dict[args.tgt]}: {source}"
    sources.append(source)
    responses.append(response)
    instructions.append(instruction)
  rubric_data = {
    "criteria":"Accuracy, Fluency, Style",
    "score1_description":"The translation contains major errors that significantly alter the meaning of the source text. It is barely comprehensible and reads like a poor machine translation. The style is completely inconsistent with the source text.",
    "score2_description":"The translation has several inaccuracies that affect the overall meaning. It is difficult to read and understand, with frequent awkward phrasings. The style only occasionally matches the source text.",
    "score3_description":"The translation is mostly accurate but has some minor errors that don't significantly alter the meaning. It is generally understandable but lacks natural flow in some parts. The style is somewhat consistent with the source text.",
    "score4_description":"The translation is accurate with only a few negligible errors. It reads naturally for the most part, with occasional minor awkwardness. The style largely matches that of the source text.",
    "score5_description":"The translation is highly accurate, conveying the full meaning of the source text. It reads as fluently as an original text in the target language. The style perfectly captures the tone and register of the source text."
  }

  score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
  final_scores = []
  for _ in range(args.turns):
    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=score_rubric,
    )
    if len(final_scores) == 0:
      final_scores = scores
    else:
      final_scores = [x + y for x, y in zip(final_scores, scores)]
  # print("Feedback:", feedbacks)
  # print("Score:", scores)
  import code; code.interact(local=locals())
  # final_scores = [score / args.turns for score in final_scores]
  # dirname = args.output_dir
  # if dirname:
  #   os.makedirs(dirname, exist_ok=True)

  # output_file = os.path.join(
  #     dirname,
  #     f"{args.src}-{args.tgt}.jsonl",
  # )
  # write_to_file(output_file, ds, final_scores, "prometheus")


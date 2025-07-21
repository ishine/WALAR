# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluates the predictions from a MetricX model."""

import dataclasses
import json
import os
from typing import Any, Tuple

from mt_metrics_eval import data
from mt_metrics_eval import stats
from mt_metrics_eval import tau_optimization
import numpy as np
import scipy.stats
import transformers


@dataclasses.dataclass
class Arguments:
    input_file: str = dataclasses.field(metadata={"help": "The input file."})

    output_file: str = dataclasses.field(
        metadata={"help": "The output file with evaluation metrics."},
    )


def _convert_to_matrices(
    instances: list[dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts the instances to metric and human score matrices."""
    system_id_to_row = {}
    segment_id_to_col = {}

    for instance in instances:
      system_id = instance["system_id"]
      segment_id = instance["segment_id"]
      if system_id not in system_id_to_row:
        system_id_to_row[system_id] = len(system_id_to_row)
      if segment_id not in segment_id_to_col:
        segment_id_to_col[segment_id] = len(segment_id_to_col)

    num_rows = len(system_id_to_row)
    num_cols = len(segment_id_to_col)
    # MTME requires that missing scores must be None, not NaN.
    metric_scores = np.full((num_rows, num_cols), None, dtype=np.dtype(object))
    human_scores = np.full((num_rows, num_cols), None, dtype=np.dtype(object))

    for instance in instances:
      system_id = instance["system_id"]
      segment_id = instance["segment_id"]
      row = system_id_to_row[system_id]
      col = segment_id_to_col[segment_id]
      metric_scores[row, col] = (
          -1 * instance["prediction"]
      )  # negate so higher is better
      human_scores[row, col] = instance["label"]
    return metric_scores, human_scores

def preprocess_dataset(input_file: str) -> list[dict[str, Any]]:
    instances = []
    if 'IndicMT' in input_file:
      with open(input_file, "r") as f:
          lines = f.readlines()
          for line in lines:
              instance = json.loads(line)
              instance['label'] = float(instance['human_score'])
            #   instance['label'] = instance['full_score']
              instance['system_id'] = instance['model']
              instances.append(instance)
    elif 'wmt' in input_file:
      with open(input_file, "r") as f:
          lines = f.readlines()
          for line in lines:
              instance = json.loads(line)
              instance['label'] = instance['score']
              instance['system_id'] = instance['system']
              instances.append(instance)
    elif 'afriMTE' in input_file:
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                instance = json.loads(line)
                instance['label'] = instance['score']
                instance['system_id'] = instance['model']
                instances.append(instance)
    return instances

def main() -> None:
    parser = transformers.HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    # import code; code.interact(local=locals())
    instances = []
    instances = preprocess_dataset(args.input_file)

    metric_seg_scores, human_seg_scores = _convert_to_matrices(instances)
    import code; code.interact(local=locals())
    metric_sys_scores = np.mean(metric_seg_scores, axis=1)

    human_sys_scores = np.apply_along_axis(
        lambda row: np.mean(row[row != None]), 1, human_seg_scores  # pylint: disable=singleton-comparison
    )

    # Segment-level correlations.
    mask = human_seg_scores.reshape(-1) != None  # pylint: disable=singleton-comparison
    metric_seg_scores = metric_seg_scores.astype(np.float32)
    metric_sys_scores = metric_sys_scores.astype(np.float32)
    human_seg_scores = human_seg_scores.astype(np.float32)
    human_sys_scores = human_sys_scores.astype(np.float32)
    seg_no_grouping_pearson, _ = scipy.stats.pearsonr(
        metric_seg_scores.reshape(-1)[mask],
        human_seg_scores.reshape(-1)[mask],
    )
    seg_no_grouping_kendalltau, _ = scipy.stats.kendalltau(
        metric_seg_scores.reshape(-1)[mask],
        human_seg_scores.reshape(-1)[mask],
    )
    tie_calib_result = tau_optimization.tau_optimization(
        metric_seg_scores.T,
        human_seg_scores.T,
        tau_optimization.TauSufficientStats.acc_23,
    )

    # System-level correlations.

    sys_pearson, _ = scipy.stats.pearsonr(human_sys_scores, metric_sys_scores)
    agree, num_pairs = stats.Agreement(human_sys_scores, metric_sys_scores)
    sys_accuracy = agree / num_pairs
    # import code; code.interact(local=locals())
    sys_spa = stats.PairwiseConfidenceError(
        human_seg_scores.reshape(-1),
        metric_seg_scores.reshape(-1),
        human_seg_scores.shape[0],
        filter_nones=True,
    )[0]

    metrics = {
        "system_level": {
            "pearson": sys_pearson,
            "accuracy": sys_accuracy,
            "spa": sys_spa,
        },
        "segment_level_no_grouping": {
            "pearson": seg_no_grouping_pearson,
            "kendalltau": seg_no_grouping_kendalltau,
        },
        "segment_level_group_by_item": {
            "accuracy": tie_calib_result.best_tau,
            "epsilon": tie_calib_result.best_threshold,
        },
    }
    print(metrics)
    import code; code.interact(local=locals())
    print(json.dumps(metrics, indent=2))

    if args.output_file:
      dirname = os.path.dirname(args.output_file)
      if dirname:
        os.makedirs(dirname, exist_ok=True)
      with open(args.output_file, "w") as out:
        out.write(json.dumps(metrics))


if __name__ == "__main__":
    main()


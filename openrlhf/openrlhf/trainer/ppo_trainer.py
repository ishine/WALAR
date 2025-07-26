import os
import time
from abc import ABC
from datetime import timedelta

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer, load_flores_dataset, get_spBLEU, remove_pad_token, zero_pad_sequences

logger = init_logger(__name__)


class BasePPOTrainer(ABC):
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        src: str = "eng",
        tgt: str = "zho_simpl",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.pretrain = pretrain
        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines

        self.prompt_split = prompt_split
        self.eval_split = eval_split
        
        self.src = src
        self.tgt = tgt  

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.remote_comet_url = self.args.remote_comet_url
        self.remote_metric_reference_url = self.args.remote_metric_reference_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Init dummy variables
        self.prompts_dataloader = None
        self.eval_dir = self.generate_kwargs['eval_dir']
        self.eval_dataloader = None
        self.max_steps = None

        self.samples_generator = None
        self.experience_maker = None
        self.remote_reward_model = None
        self.remote_reward_model2 = None
        self.remote_comet = None
        self.remote_metric_reference = None

        if self.args.agent_func_path:
            from openrlhf.trainer.ppo_utils.experience_maker_async import SamplesGeneratorAsync as SamplesGenerator
        else:
            from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
            self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="reload_states")

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                self.actor_model_group.async_run_method(method_name="offload_states")

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()
        # breakpoint()
        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dir is not None:
            self.evaluate(self.eval_dir, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def evaluate(self, eval_dir, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        import sacrebleu
        two2three = {
            "en": "eng",
            "zh": "zho_simpl",
            "sw": "swh",
            "ta": "tam",
            "fr": "fra",
            "bn": "bng",
            "fi": "fin",
        }
        lang_dict = {
            'eng': "English",
            "zho_simpl": "Chinese",
            'swh': "Swahili",
            "tam": "Tamil",
            "fra": "French",
            "bng": "Bengali",
            "fin": "Finnish",
        }
        src_lang = two2three[self.src]
        val_lang_list = [two2three[self.tgt]]
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # breakpoint()
        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
        for tgt_lang in val_lang_list:
            src_dataset, tgt_dataset = load_flores_dataset(eval_dir, f"{src_lang}-{tgt_lang}")
            assert len(src_dataset) == len(tgt_dataset), "Source and target datasets must have the same length."
            n = len(src_dataset)
            src_dataset, tgt_dataset = src_dataset[:n//2], tgt_dataset[:n//2]  # Use only the first half for evaluation
            with torch.no_grad():
                # First collect all prompts and labels
                all_prompts = []
                all_labels = []
                prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources
                for src_text, tgt_text in zip(src_dataset, tgt_dataset):
                    prompt = f"{src_text}\nTranslate from {lang_dict[src_lang]} to {lang_dict[tgt_lang]}:\n"
                    message = [
                        {"role": "user", "content": prompt},
                    ]
                    if 'Qwen3' in self.pretrain:
                        new_prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    else:
                        new_prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                    all_prompts.append(new_prompt)
                    all_labels.append(tgt_text)
                # for datasources, prompts, labels in eval_dataloader:
                #     all_prompts.extend(prompts)
                #     all_labels.extend(labels)
                #     # Create mapping for each prompt to its corresponding data source
                #     for prompt, datasource in zip(prompts, datasources):
                #         prompt_to_datasource[prompt] = datasource

                # Generate samples and calculate rewards
                generate_kwargs = self.generate_kwargs.copy()
                generate_kwargs["temperature"] = temperature
                generate_kwargs["n_samples_per_prompt"] = 1
                
                if self.remote_comet:
                    samples_list1 = self.samples_generator.generate_samples(
                        all_prompts, all_labels, remote_reward_model=self.remote_comet, **generate_kwargs
                    )
                if self.remote_metric_reference:
                    samples_list2 = self.samples_generator.generate_samples(
                        all_prompts, all_labels, remote_reward_model=self.remote_metric_reference, **generate_kwargs
                    )
                # experiences = self.experience_maker.make_experience(samples_list1)
                # samples = [self.tokenizer.batch_decode(experiences[i].sequences[0].unsqueeze(0), skip_special_tokens=True)[0] for i in range(len(experiences))]
                # samples = [sample.split(f"Translate from {lang_dict[src_lang]} to {lang_dict[tgt_lang]}:")[1].strip() for sample in samples]
                sequences_list = [s.sequences for s in samples_list1]
                attention_mask_list = [s.attention_mask for s in samples_list1]
                samples = sum(
                    [
                        self.tokenizer.batch_decode(remove_pad_token(seq, attention_mask), skip_special_tokens=False)
                        for seq, attention_mask in zip(sequences_list, attention_mask_list)
                    ],
                    [],
                )
                # print(samples)
                # print(all_labels)
                samples = [query.split('<|im_start|>assistant\n', 1)[1].split("<|im_end|>", 1)[0].strip() for query in samples]
                bleu = get_spBLEU(samples, all_labels)
        
                logs = {}
                # Reshape rewards to (num_prompts, n_samples_per_prompt)
                if self.remote_comet:
                    rewards_list1 = [sample.rewards for sample in samples_list1]
                    rewards1 = torch.tensor(rewards_list1).reshape(-1, n_samples_per_prompt)
                    logs[f"{lang_dict[src_lang]}-{lang_dict[tgt_lang]}-comet_reward"] = rewards1.mean().item()
                if self.remote_metric_reference:
                    rewards_list2 = [sample.rewards for sample in samples_list2]
                    rewards2 = torch.tensor(rewards_list2).reshape(-1, n_samples_per_prompt)
                    logs[f"{lang_dict[src_lang]}-{lang_dict[tgt_lang]}-metric_reference_reward"] = rewards2.mean().item()

                # Calculate global averages
                logs[f"{lang_dict[src_lang]}-{lang_dict[tgt_lang]}-bleu"] = bleu
                # Log to wandb/tensorboard
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, global_step)
                print(f"Evaluation for {lang_dict[src_lang]}-{lang_dict[tgt_lang]}: Done!")

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=self.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=self.eval_split,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )

    def get_max_steps(self):
        return self.max_steps


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        src: str = "eng",
        tgt: str = "zho_simpl",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            src,
            tgt,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)
            
        if self.args.remote_comet_url:
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel
            self.remote_comet = RemoteRewardModel.remote(self.args, self.remote_comet_url)
        
        if self.args.remote_metric_reference_url:
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_metric_reference = RemoteRewardModel.remote(self.args, self.args.remote_metric_reference_url)
            
        if self.args.remote_rm_url2:
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model2 = RemoteRewardModel.remote(self.args, self.args.remote_rm_url2)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)
        # self.evaluate(self.eval_dir, steps, temperature=0.0, n_samples_per_prompt=1)
        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
            )

            filtered_samples = []
            number_of_samples = 0
            for _, rand_prompts, labels in self.prompts_dataloader:
                print(f"len(rand_prompts): {len(rand_prompts)}, len(labels): {len(labels)}")
                print(f"rand_prompts: {rand_prompts[0]}")
                remote_reward_model = self.remote_reward_model
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts, labels, remote_reward_model=remote_reward_model, remote_reward_model2=self.remote_reward_model2, **self.generate_kwargs
                )
                reward1_list = [sample.info.get('reward1', 0) for sample in rollout_samples]
                reward2_list = [25 * sample.info.get('reward2', 0) for sample in rollout_samples]
                rule_penalty_percent = rollout_samples[0].info.get('rule_penalty_percent', 0)
                lang_penalty_percent = rollout_samples[0].info.get('lang_penalty_percent', 0)
                truncate_percent = rollout_samples[0].info.get('truncate_percent', 0)
                pbar.update()
                # breakpoint()
                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0

                experiences = self.experience_maker.make_experience_batch(rollout_samples)
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                )
                print(f"sample0: {sample0}")
                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
                if self.critic_model_group is not None:
                    refs.extend(
                        self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    )
                ray.get(refs)
                # breakpoint()
                status = self.ppo_train(steps)
                # breakpoint()
                status['reward1'] = sum(reward1_list) / len(reward1_list) if reward1_list else 0.0
                status['reward2'] = sum(reward2_list) / len(reward2_list) if reward2_list else 0.0
                status['rule_penalty_percent'] = rule_penalty_percent
                status['lang_penalty_percent'] = lang_penalty_percent
                status['truncate_percent'] = truncate_percent
                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {steps}: {status}")
                status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

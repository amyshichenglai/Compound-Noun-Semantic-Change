from typing import Optional, Any, Tuple
from argparse import ArgumentParser
import logging
import os
import json
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForMaskedLM


from data import CorpusData, END_SEQUENCE, Sentence, MAX_SEQ_LENGTH


IGNORE_LOSS_INDEX = -100 # see https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
MAX_STORED_CHECKPOINTS = 3 # beyond this, oldest are removed

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        help="the name of an existing model e.g. 'bert-base-german-cased'")
    parser.add_argument("--output_dir", type=str, help="location to write logs, fine tuned model, etc to.")
    parser.add_argument("--cached_tokenizer_path_or_name", type=str,
                        help="path to dir with cached fine-tuned tokenizer "
                             "OR the name of an existing model e.g. 'bert-base-german-cased'")
    parser.add_argument("--cached_inputs", type=str, help=".pickle file with vectorized input data")
    parser.add_argument("--uncased", action='store_true', help="lowercase all inputs")
    parser.add_argument("--text_mode", choices=["lemma", "text"], required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="set to positive value to break early from training"
                             "after going through however many examples")
    parser.add_argument("--warmup_proportion", type=float, default=0.0, help="float from 0-1 specifying proportion"
                                                                             "of max_train_steps to be used for warmup steps")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=0, help="Number of steps after which to checkpoint the model "
                                                                  "(when training)")
    parser.add_argument("--save_per_epoch", action='store_true', help="if set, will save a checkpoint after each epoch")
    parser.add_argument("--train_checkpoint", type=str, help="directory with checkpoint to resume from")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_seed", type=int, default=2666)

    parser.add_argument("--mask_percentage", type=float, default=0.15,
                        help="float from 0.0 to 1.0 describing proportion "
                             "of tokens that should be masked in MLM setup. 0.15"
                             " is the default from the BERT paper.")
    # misc args
    parser.add_argument("--dry_run", action='store_true', help="load everything but don't take any action!"
                                                               " no writing of any kind, and exiting early")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to attempt to use")

    args = parser.parse_args()

    cased_log_name = "uncased" if args.uncased else "cased"
    # split_log_name = "split" if args.split_compounds else "unsplit"
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%d/%m/%Y %H:%M:%S",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(
                            filename=f"{args.output_dir}/sem-change-clustering_domain_adaptation_{cased_log_name}_.log",
                            encoding='utf-8',
                            mode='a+')])

    logging.info(f"args: {args}")

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu':
        logging.warning("Running on cpu - please be sure this is desired")
    else:
        logging.info(f"running on cuda:{args.gpu}")
    dataset = None
    # resolve location of cached dataset

    # be very explicit about where the model and the tokenizer are loaded from:
    # tokenizer = None
    tokenizer = AutoTokenizer.from_pretrained(args.cached_tokenizer_path_or_name)
    logging.info(f"tokenizer loaded from {args.cached_tokenizer_path_or_name}")
    # if os.path.exists(os.path.join(args.cached_tokenizer_path_or_name, "tokenizer.json")):
    #     tokenizer = AutoTokenizer.from_pretrained(args.cached_tokenizer_path_or_name)
    #     logging.info(f"tokenizer loaded from {args.cached_tokenizer_path_or_name}")
    # else:
    #     tokenizer = get_config(args.cached_model_path_or_name)['tokenizer']
    #     logging.warning(f"tokenizer loaded from *MODEL* config: {args.cached_model_path_or_name}\n"
    #                     f"This is expected only if loading from pre-trained model")

    cached_input_dir = args.output_dir
    print(f"looking for cached inputs at {args.cached_inputs}")
    if os.path.exists(args.cached_inputs):
        dataset = CorpusData(
            cached_data=args.cached_inputs,
        )
        # if we need to hang onto the (nice, python, readable, etc) Sentence portion
        # of the cached data, grab it here before it's removed!
        dataset.make_dataloader_friendly()
        logging.info(f"loaded CorpusData from cached copy at {args.cached_inputs}")
    else:
        raise ValueError("Need to provide a cached inputs file (that can be loaded)")

    if not args.output_dir:
        raise ValueError("Need to provide an output dir to train embeddings")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    logging.info(f"loaded model from {args.model_name}")
    logging.info("Starting training")
    trainer = EmbeddingsTrainer(
        args,
        dataset,
        model,
        tokenizer=tokenizer,
        eval_dataset=None
    )
    if args.dry_run:
        logger.info(f"dry run enabled: stopping before training")
        return
    steps_trained, avg_loss = trainer.train_embeddings()
    logger.info(f"trained {steps_trained} global steps, avg loss: {avg_loss}")
    trainer.save_model()


class EmbeddingsTrainer:
    def __init__(self, args,
                 dataset: Dataset,
                 pretrained_model,
                 tokenizer,
                 eval_dataset: Optional[Dataset] = None
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = dataset
        self.train_dataloader = self._get_train_dataloader(self.train_dataset)
        self.eval_dataset = eval_dataset
        # if it's any consolation, this is an often-copied snippet
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in pretrained_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in pretrained_model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.model = pretrained_model
        self.model.to(self.args.device)
        if args.max_train_steps > 0:
            total_steps = args.max_train_steps
        else:
            total_steps = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=args.adam_epsilon,
        )

        warmup_steps = int(self.args.max_train_steps * self.args.warmup_proportion)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        if (
          self.args.train_checkpoint
          and os.path.isfile(os.path.join(self.args.train_checkpoint, "optimizer.pt"))
          and os.path.isfile(os.path.join(self.args.train_checkpoint, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(self.args.train_checkpoint, "optimizer.pt")))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.args.train_checkpoint, "scheduler.pt")))
            logging.info(f"loaded optimizer and scheduler from {args.train_checkpoint}")

        # check if COHA END_SEQUENCE is in tokenizer as a single unit
        end_seq_toks = self.tokenizer(END_SEQUENCE)['input_ids'][1:-1]
        # (removing cls and sep)
        if len(end_seq_toks) == 1:
            self.coha_end_sequence_id = end_seq_toks[0]
        else:
            self.coha_end_sequence_id = None


    def train_embeddings(self):

        total_steps = self.args.num_epochs * len(self.train_dataset)

        logging.info(f"Running training:")
        logging.info(f" Num batches = {len(self.train_dataloader)}")
        logging.info(f" Num Epochs = {self.args.num_epochs}")
        logging.info(f" Batch size per device = {self.args.batch_size}")
        logging.info(f" Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logging.info(f" Total optimization steps = {total_steps}")


        global_steps = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # check if we're resuming
        if self.args.train_checkpoint and os.path.exists(self.args.train_checkpoint):
            dir_name = os.path.basename(os.path.normpath(self.args.train_checkpoint))
            if dir_name.startswith("checkpoint-"):
                global_steps = int(dir_name.split('checkpoint-')[-1])
                epochs_trained = global_steps // len(self.train_dataloader) // self.args.gradient_accumulation_steps
                steps_trained_in_current_epoch = global_steps % len(self.train_dataloader) // self.args.gradient_accumulation_steps
                logging.info(f"resuming training after {global_steps}, after {epochs_trained} epochs"
                             f" plus {steps_trained_in_current_epoch} steps in current epoch")
            else:
                raise ValueError("trying to start from checkpoint without correct checkpoint- directory")

        # set up iter variables


        tr_loss = 0.0

        self._set_seed(self.args.train_seed)
        self.model.zero_grad()
        train_itr = trange(self.args.num_epochs, desc="Epochs_itr")
        for epoch in train_itr:
            if epoch < epochs_trained:
                continue
            step_itr = tqdm(self.train_dataloader, desc="step_itr")
            for step, batch in enumerate(step_itr):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                inputs, labels = self._hf_torch_mask_tokens(batch)
                inputs = inputs.to(self.args.device)
                labels = labels.to(self.args.device)
                self.model.train()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad()
                    global_steps += 1
                    # save checkpoint
                    if self.args.save_steps > 0 and global_steps % self.args.save_steps == 0:
                        self.save_model(global_steps)
                    # break out of both loops if global maximum steps have been reached:
                    if self.args.max_train_steps and global_steps > self.args.max_train_steps:
                        break
            if self.eval_dataset is not None:
                self.eval_embeddings(self.eval_dataset)
            if self.args.save_per_epoch:
                self.save_model(global_steps)
            # break out of epoch loop if global max steps reached
            if self.args.max_train_steps and global_steps > self.args.max_train_steps:
                break

        return global_steps, tr_loss / global_steps


    def eval_embeddings(self, dataset):
        dataloader = self._get_eval_dataloader(dataset)
        eval_loss = 0.0
        eval_steps = 0
        self.model.eval()

        for batch in tqdm(dataloader, desc="Eval"):
            eval_loss += self._eval_step(batch).mean().item()
            eval_steps += 1

        eval_loss = eval_loss / eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity.cpu().numpy().tolist()}
        # write this as a json file
        eval_path = os.path.join(self.args.output_dir, "eval")
        os.makedirs(eval_path, exist_ok=True)
        with open(os.path.join(eval_path, "eval_results.json"), 'w', encoding='utf-8') as eval_out:
            json.dump(result, eval_out, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"Eval result: {json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)}")
        return result

    def _eval_step(self, batch):
        # inputs, labels = self._mask_tokens(batch)
        inputs, labels = self._hf_torch_mask_tokens(batch)
        inputs = inputs.to(self.args.device)
        labels = labels.to(self.args.device)
        with torch.no_grad():
            outputs = self.model(inputs, labels=labels)
            lm_loss = outputs.loss
            return lm_loss

    def save_model(self, steps_trained: Optional[int]=None):
        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        # Create output directory if needed
        prefix = "" if steps_trained is None else f"checkpoint-{steps_trained}"
        output_dir = os.path.join(self.args.output_dir, prefix)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        try:
            self.tokenizer.save_pretrained(output_dir)
        except:
            pass

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # if this is a midway checkpoint also save optimizer / scheduler state:
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # check to remove old checkpoint
        checkpoints = sorted(
            [int(dirname.split('-')[1])
             for dirname in os.listdir(output_dir)
             if dirname.startswith('checkpoint')]
        )
        checkpoint_numbers_to_remove = []
        if len(checkpoints) > MAX_STORED_CHECKPOINTS:
            checkpoint_numbers_to_remove = checkpoints[:-MAX_STORED_CHECKPOINTS]
        for checkpoint_num_to_remove in checkpoint_numbers_to_remove:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{checkpoint_num_to_remove}")
            for checkpt_f in os.listdir(checkpoint_dir):
                os.remove(os.path.join(checkpoint_dir, checkpt_f))
            os.rmdir(checkpoint_dir)


# TODO: think about 'whole word masking' -> I dunno, it might make sense in conjuntion w/ splitting up all compounds
#       into separate words
    # this basically comes from https://github.com/huggingface/transformers/blob/7ec1dc8817a99d16e6f9e0ab94ce4027ef74b72d/src/transformers/data/data_collator.py#L742

    def _hf_torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.args.mask_percentage)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            # deal with the additional non-sequitur token that occurs in COHA data (where @ obfuscated tokens were removed)
            if self.coha_end_sequence_id:
                for i in range(len(special_tokens_mask)):
                    additional_mask = [1 if elt == self.coha_end_sequence_id else 0 for elt in special_tokens_mask[i]]
                    for j in range(MAX_SEQ_LENGTH):
                        special_tokens_mask[i][j] += additional_mask[j]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset, sampler=RandomSampler(dataset), batch_size=self.args.batch_size
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=self.args.eval_batch_size
        )


if __name__ == "__main__":
    main()

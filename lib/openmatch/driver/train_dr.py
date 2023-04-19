import logging
import os
import sys
import pdb
from pathlib import Path
import deepspeed
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
# from tensorboardX import GlobalSummaryWriter as SummaryWriter
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from lib.openmatch.arguments import DataArguments
from lib.openmatch.arguments import DenseTrainingArguments as TrainingArguments
from lib.openmatch.arguments import ModelArguments
from lib.openmatch.dataset import QPCollator, TrainDataset, EvalDataset
from lib.openmatch.modeling import DenseModel
from lib.openmatch.trainer import DenseTrainer
from lib.openmatch.trainer import GCDenseTrainer
from lib.openmatch.utils import is_rank_0
from lib.openmatch.utils import load_stuff

logger = logging.getLogger(__name__)

# the following environment varialbes are necessary to have 
# for torch.distributed.launch, but maybe not for deepspeed launch...
# the last one NCCP_P2P_DISABLE was particularly important...
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['NCCL_DEBUG']='INFO'
# os.environ['NCCL_DEBUG_SUBSYS']='ALL'
# os.environ['NCCL_IB_DISABLE']='1'
# os.environ['NCCL_P2P_DISABLE']='1' ### needed to add this to get it to work on workstation with A6000. dont ask me why
# os.environ['NCCL_SOCKET_IFNAME']='eth0' # 'lo'### ultimately did not need this. 
# os.environ["NCCL_SOCKET_IFNAME"]="ens5" ### for more than 2 nodes?

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Lr decay while iterative training with hard negative
    for iter in range(model_args.iter_num):
        training_args.learning_rate*=model_args.decay_rate

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16 training: %s, bf16 training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    ### do not do either of these if you're launching with MPI scripts (run_multi_node.sh)
    # deepspeed.init_distributed('nccl')
    # torch.distributed.init_process_group("nccl")

    num_labels = 1
    config, tokenizer = load_stuff(model_args, data_args)
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
    #                                cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    # train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)
    train_dataset = TrainDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    eval_dataset = EvalDataset(tokenizer, data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.eval_path is not None else None
    print(f"length of training dataset: {len(train_dataset)} examples")
    # pdb.set_trace()

    ### set up tensorboard logging
    if is_rank_0():
        tbWriter = SummaryWriter(training_args.logging_dir)
        tb_callback = TensorBoardCallback(tbWriter)

    trainer_cls = GCDenseTrainer if training_args.grad_cache else DenseTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        callbacks=[tb_callback] if is_rank_0() else []
    )
    train_dataset.trainer = trainer

    if not training_args.zero_shot_eval:
        trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

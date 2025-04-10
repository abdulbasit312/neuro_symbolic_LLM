import argparse
import torch
import os
import wandb
import random
import json
import torch.multiprocessing as mp
from torch.optim import AdamW
import time

from utils.constraints import ConstraintManager
from models.loco.model import QA
from models.loco.trainer import Trainer
from torch.distributed import init_process_group, destroy_process_group
from models.loco.trainer import DEFAULT_CONFIG
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/w/331/abdulbasit/loco-llm/assets"

torch.autograd.set_detect_anomaly(True)

def ddp_setup(rank, world_size:int, port:int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# =================================================

def main(rank:int, world_size:int, config:object, wandb_run:object, run_parallel:bool, port:int=12355):
    
    torch.manual_seed(1337)
    random.seed(1337)

    if run_parallel: ddp_setup(rank, world_size=world_size, port=port)
    if "quantization" not in config: config["quantization"] = False

    model = QA(
        quantization=config["quantization"],
        model_hf_name=config["model"], 
        factuality=config["factuality"] if "factuality" in config else DEFAULT_CONFIG["factuality"],
        use_table_truth=config["use_table_truth"] if "use_table_truth" in config else DEFAULT_CONFIG["use_table_truth"],
        gpu_id=rank
    )

    # Training utilities
    constraints = ConstraintManager()
    optimizer = AdamW(model.parameters(), lr = config["lr"])

    # Load model
    if "checkpoint" in config:
        print(f"[-] Loading LoRA adapter: {config['checkpoint']}")
        model.model.load_adapter("/scratch/expires-2025-Apr-10/abdulbasit/cktp/checkpoints/"+config['checkpoint'],adapter_name="dddde")
    
    model.to(rank)

    print("[-] Running trainer...")
    trainer = Trainer(
        model=model,
        lr=config["lr"],
        constraint_mg=constraints, 
        wandb=wandb_run,
        checkpoints_path=os.path.join("checkpoints"),
        optimizer=optimizer,
        gpu_id=rank,
        config=config,
        run_parallel=run_parallel,
        val_interval=config["val_interval"]
    )

    # Training mode
    if config["task"] == "train":
        trainer.run_train(mode=config["training"])
    # Eval
    elif config["task"] == "eval":
        start = time.time()
        trainer.run_eval()
        end = time.time()
        print(f"[-] Time elapsed: {end-start}s")
    else:
        raise Exception("Invalid task seleted.")

    # Cleanup
    del model
    del optimizer
    del constraints
    del trainer
    torch.cuda.empty_cache()
    if run_parallel: destroy_process_group()

# =================================================

if __name__ == '__main__':

    # Configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--constraint_type', type=str, choices=["implication", "inverse_implication", "negation", "all"], required=True)
    parser.add_argument('--port', type=str, default=12355)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--quantization', type=bool)
    parser.add_argument('--lr_scheduler', type=bool)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Wandb setup
    if config["wandb"] == True:
        wandb.login()
        wandb_run = wandb.init(
            project="large-semantic-language-models",
            name=args.run_name,
            config={
                "model": config["model"],
                "learning_rate": config["lr"],
                "batch_size": config["batch_size"],
                "accumulation_steps": config["accumulation_steps"],
                "epochs": config["epochs"]
            }
        )
    else: wandb_run = None

    if args.constraint_type is not None:
        config["constraint_type"] = args.constraint_type

    if args.model is not None:
        config["model"] = args.model

    if args.quantization is not None:
        config["quantization"] = args.quantization

    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint

    if args.lr_scheduler is not None: config["lr_scheduler"] = True
    else: config["lr_scheduler"] = False

    # Start in parallel
    if config["parallel"] is not True:
        print("[-] Running in single instance...")
        main(rank=0, world_size=1, config=config, wandb_run=wandb_run, run_parallel=False, port=args.port)
    else:
        print("[-] Running on parallel instance...")
        ngpus = torch.cuda.device_count()
        mp.spawn(main, args=(ngpus, config, wandb_run, True, args.port), nprocs=ngpus)
    

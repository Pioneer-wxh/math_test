#  ------------------------------------------------------------------------------------------
#  SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models
#  arXiv: https://arxiv.org/abs/2409.00055
#  Copyright (c) 2024 Yang Cao
#  Licensed under the Apache License, Version 2.0.
#  ------------------------------------------------------------------------------------------
import numpy as np
from datasets import load_dataset
from hf_to_rwkv import convert_to_rwkv
import pathlib
import torch
from loralib import LoRAModel, LoRAConfig
from sorsa import SORSAModel, SORSAConfig, SORSATrainer, SORSATrainingArguments
from dashlib import LoRADashModel, LoRADashConfig#修改
from fossvlib import FOSSVModel,FOSSVConfig#修改
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    Trainer,
    TrainingArguments,)
from dataset import (
    preprocess_metamathqa,
    preprocess_codefeedback,
    preprocess_codefeedback_instructed,
    collate_fn,
)
from test import (
    test_gsm,
    test_math,
    test_gsm_rwkv,
    test_math_rwkv,
    test_human_eval,
    test_human_eval_rwkv,
)
import argparse
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from dotenv import dotenv_values
from huggingface_hub import login

CONFIG = {"lora": LoRAConfig, "pissa": LoRAConfig, "sorsa": SORSAConfig,"milora":LoRAConfig,"loradash":LoRADashConfig,"fossv":FOSSVConfig}#需要增添修改milora，fossv，lora-dash
MODEL = {"lora": LoRAModel, "pissa": LoRAModel, "sorsa": SORSAModel,"milora":LoRAModel,"loradash":LoRADashModel,"fossv":FOSSVModel}
PREFIX = {"lora": "lora_", "pissa": "lora_", "sorsa": "sorsa_","milora":"lora_","loradash":"loradash_","fossv":"fossv_"}

parser = argparse.ArgumentParser(
    prog="SORSA Runner",
    description="Train and Test SORSA Layers",
    epilog="--help for more information",
)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("-p", "--peft", choices=["sorsa", "lora", "pissa","milora","loradash","fossv"])  # 选择适配器类型 #需要增添修改

parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--tokenizer", type=str, default=None)

parser.add_argument("--mod", type=str, default="small")#修改

parser.add_argument("-n", "--name", type=str, default="SORSA")
parser.add_argument("-r", "--rank", type=int, default=4)
parser.add_argument("-l", "--length", type=int, default=512)
parser.add_argument("-a", "--alpha", type=float, default=None)
#parser.add_argument("-g", "--gamma", type=float, default=4e-4)修改
parser.add_argument("-d", "--dropout", type=float, default=0.0)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-T", "--test", action="store_true")
parser.add_argument("-m", "--merge", action="store_true")

parser.add_argument("--train-dataset", choices=["metamath", "code"], default="metamath")
parser.add_argument("--split", type=str, default="")

parser.add_argument(
    "--test-dataset", choices=["gsm-8k", "math", "humaneval"], default="gsm-8k"
)
parser.add_argument(
    "--test-precision", choices=["bf16", "fp16", "fp32"], default="bf16"
)

parser.add_argument("--rwkv", action="store_true")

parser.add_argument(
    "--mix-precision", choices=["bf16", "fp16", "fp32", "tf32"], action="append"
)
parser.add_argument("--grad-cp", type=float, default=1.0)
parser.add_argument("--scheduler", type=str, default="cosine")
parser.add_argument("--warmup", type=float, default=0.03)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--valid-batch-size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--accum-steps", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--wd", type=float, default=0.00)

parser.add_argument("--svd-cache-path", type=pathlib.Path, default="./svd_cache")
parser.add_argument("--run-path", type=pathlib.Path, default="./runs")
parser.add_argument("--cache-path", type=pathlib.Path, default="./cache")

parser.add_argument("--local", action="store_true")


args = parser.parse_args()


class TrainerConfig:
    def __init__(self, args: argparse.Namespace):
        self.seed = args.seed
        torch.manual_seed(args.seed)
        self.rank = args.rank
        self.dropout = args.dropout
       # self.gamma = args.gamma修改
        self.batch_size = args.batch_size
        self.valid_batch_size = args.valid_batch_size
        self.num_epochs = args.epochs
        self.accum_steps = args.accum_steps
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")
        self.run_path = f"{args.run_path}/{args.name}"
        self.metadata_path = f"{self.run_path}/metadata.pt"
        self.checkpoint_path = f"{self.run_path}/checkpoint"
        if args.local is False:
            env = dotenv_values(".env")
            login(token=env["hf"])
        if args.tokenizer is None:
            args.tokenizer = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, cache_dir=args.cache_path, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if "metamath" in args.train_dataset:
            self.train_dataset = load_dataset(
                "meta-math/MetaMathQA", split=f"train{args.split}"
            )
            self.train_dataset = self.train_dataset.map(
                lambda x: preprocess_metamathqa(x, self.tokenizer, args.length)
            )
        if "code" in args.train_dataset:
            self.train_dataset = load_dataset(
                "m-a-p/CodeFeedback-Filtered-Instruction", split=f"train{args.split}"
            )
            self.train_dataset = self.train_dataset.map(
                lambda x: preprocess_codefeedback_instructed(
                    x, self.tokenizer, args.length
                )
            )
        self.train_dataset.set_format(
            type="torch", columns=["input_ids", "labels", "attention_mask"]
        )
        if args.rwkv:
            target = ["attention", "feed_forward"]
        else:
            target = [
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        # self.config = CONFIG[args.peft](#可能修改
        #     base_model_name_or_path=args.model,
        #     target_modules=target,
        #     rank=self.rank,
        #     alpha=args.alpha,
        #     dropout=self.dropout,
        # )
        # 1. 创建通用参数字典（所有适配器共有的参数）
        config_args = {
            "base_model_name_or_path": args.model,
            "target_modules": target,
            "rank": self.rank,
            "alpha": args.alpha,
            "dropout": self.dropout,
        }

        # 2. 仅在适配器类型是 sorsa 时添加 mod 参数
        if args.peft == "fossv":
            config_args["mod"] = args.mod

        # 3. 动态传递参数（其他适配器不会收到 mod）
        self.config = CONFIG[args.peft](**config_args)


        # Load model in ALL float32 (residual is quantized to bf16)
        if os.path.isdir(args.svd_cache_path):
            print("Loading SVDed Model...")
            self.model = MODEL[args.peft].from_pretrained(
                pretrained_model_name_or_path=args.svd_cache_path,
            )
        else:
            print("Start from Pretrained Model...")#修改
            self.model = MODEL[args.peft](self.config)#如果没有就从huggingface上加载，并且下载到svd这个文件夹下
            self.model.to(self.device)
            if "sorsa" in args.peft:
                self.model.sorsa_init(
                    weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )
                self.model.save_pretrained(args.svd_cache_path)#同时也储存init初始化后的模型也就是说覆盖
            
            if args.peft == "lora":
                self.model.lora_init(
                    pissa=False,milora=False, weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )

            # 如果使用的是pissa适配器，则进行pissa初始化
            if "pissa" in args.peft:
                self.model.lora_init(
                    pissa=True,milora=False, weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )
                self.model.save_pretrained(args.svd_cache_path)  # 保存模型到指定路径
            
            if "milora" in args.peft:
                self.model.lora_init(
                    pissa=False,milora=True, weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )
                self.model.save_pretrained(args.svd_cache_path)  # 保存模型到指定路径
            
            if "loradash" in args.peft:
                self.model.loradash_init(
                     weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )
            
            if "fossv" in args.peft:
                self.model.fossv_init(
                     weight_dtype=torch.bfloat16, adapter_dtype=torch.float32
                )
                self.model.save_pretrained(args.svd_cache_path)  # 保存模型到指定路径
        self.model.train()
        self.model.set_trainable(True)

        self.training_arguments = TrainingArguments(#修改，去掉了前缀，SORSA
            run_name=args.name,
            output_dir=self.run_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accum_steps,
            logging_dir=self.run_path,
            logging_steps=1,
            save_steps=1000,
            save_total_limit=1,
            report_to=["wandb"],
            save_strategy="steps",
            lr_scheduler_type=args.scheduler,
            warmup_ratio=args.warmup,
            seed=self.seed,
            weight_decay=args.wd,
            learning_rate=args.lr,
            max_grad_norm=args.grad_cp,
           # gamma=self.gamma,修改
            bf16="bf16" in args.mix_precision,
            fp16="fp16" in args.mix_precision,
            tf32="tf32" in args.mix_precision,
            remove_unused_columns=False,  # 添加此行  #修改，如果去掉了SORSA，就要加上这行代码
        )


if args.test and not args.rwkv:
    if args.local is False:
        env = dotenv_values(".env")
        login(token=env["hf"])
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.cache_path, use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/checkpoint"
    precision_mapping = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    if "math" in args.test_dataset:
        test_math(
            model=checkpoint_path,
            tokenizer=args.tokenizer,
            precision=precision_mapping[args.test_precision],
        )
    if "gsm-8k" in args.test_dataset:
        test_gsm(
            model=checkpoint_path,
            tokenizer=args.tokenizer,
            precision=precision_mapping[args.test_precision],
        )
    if "humaneval" in args.test_dataset:
        test_human_eval(
            model=checkpoint_path,
            tokenizer=args.tokenizer,
            precision=precision_mapping[args.test_precision],
        )

if args.test and args.rwkv:
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/pytorch_model.pth"
    if "math" in args.test_dataset:
        test_math_rwkv(
            model=checkpoint_path,
            precision=args.test_precision,
        )
    if "gsm-8k" in args.test_dataset:
        test_gsm_rwkv(
            model=checkpoint_path,
            precision=args.test_precision,
        )
    if "humaneval" in args.test_dataset:
        test_human_eval_rwkv(
            model=checkpoint_path,
            precision=args.test_precision,
        )

elif args.train:
    config = TrainerConfig(args)
    trainer = Trainer(#修改 SORSATrainer改为Trainer
        model=config.model,
        args=config.training_arguments,
        data_collator=lambda x: (collate_fn(x, config.tokenizer)),
        train_dataset=config.train_dataset,
    )
    print(f"Trainable Parameters: {trainer.get_num_trainable_parameters()}")
    trainer.train()
    trainer.save_model(config.checkpoint_path)

elif args.merge and not args.rwkv:
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/checkpoint"
    model = MODEL[args.peft].from_pretrained(checkpoint_path,local_files_only=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.merge(True)
    for name, param in model.named_parameters():
        if PREFIX[args.peft] in name or "weight_u_top" in name or "weight_vt_top" in name:#修改
            model._set_submodule(name, None)
    for name, _ in model.named_buffers():# 修改
        if "U" in name or "S" in name or "Vt" in name:
            model._set_submodule(name, None)
    model.model.save_pretrained(checkpoint_path)
    if args.local is False:
        env = dotenv_values(".env")
        login(token=env["hf"])
    config = AutoConfig.from_pretrained(args.model)
    config.save_pretrained(checkpoint_path)

elif args.merge and args.rwkv:
    run_path = f"{args.run_path}/{args.name}"
    checkpoint_path = f"{run_path}/checkpoint"
    model = MODEL[args.peft].from_pretrained(checkpoint_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.merge(True)
    for name, param in model.named_parameters():
        if PREFIX[args.peft] in name or "weight_u_top" in name or "weight_vt_top" in name:#修改
            model.set_submodule(name, None)
    for name, _ in model.named_buffers():# 修改
        if "U" in name or "S" in name or "Vt" in name:
            model._set_submodule(name, None)
    pth = convert_to_rwkv(model.model.state_dict())
    torch.save(pth, f"{run_path}/pytorch_model.pth")

import torch
from transformers import AutoModel,AutoTokenizer
import copy
import argparse
import os
import shutil
import glob
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--input_model_path", type=str)
    parser.add_argument("--output_model_path", type=str)
    parser.add_argument("--num_layers", type=int, default=12)
    args = parser.parse_args()

    original_model = AutoModel.from_pretrained(args.model_name_or_path,cache_dir=args.input_model_path)
    original_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,cache_dir=args.input_model_path)
    original_model.save_pretrained(args.input_model_path)
    original_tokenizer.save_pretrained(args.input_model_path)
    state_dict = original_model.state_dict()
    keys = state_dict.keys()
    new_state_dict = copy.deepcopy(state_dict)

    for i in tqdm(range(args.num_layers)):
        new_state_dict[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
        new_state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'] /= 10
        new_state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] /= 10

        new_state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] /= 100
        new_state_dict[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
        new_state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'] /= 10
        new_state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] /= 10
    new_state_dict['shared.weight'] /= 100

    os.makedirs(args.output_model_path, exist_ok=True)
    torch.save(new_state_dict, os.path.join(args.output_model_path, "pytorch_model.bin"))

    # copy other files
    files = glob.glob(os.path.join(args.input_model_path, "*"))
    for file in files:
        if file != os.path.join(args.input_model_path, "pytorch_model.bin"):
            shutil.copy(file, args.output_model_path)

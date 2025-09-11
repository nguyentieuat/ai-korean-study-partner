# %%writefile /content/train_topik_gen_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_TRAIN = "/content/topik_mcq_sft.train.jsonl"   # <== nhớ đổi nếu bạn để chỗ khác
DATA_VAL   = "/content/topik_mcq_sft.val.jsonl"
OUT = "/content/drive/MyDrive/ckpts/topik_gen/llama3-8b/adapter"  # lưu adapter trên Drive

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

# Khuyến nghị: thêm SDPA để tiết kiệm VRAM nếu có
try:
    model.config.attn_implementation = "sdpa"
except Exception:
    pass

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

ds_train = load_dataset("json", data_files=DATA_TRAIN, split="train")
ds_val   = load_dataset("json", data_files=DATA_VAL,   split="train")

def fmt(ex):  # mỗi dòng đã là messages=[system,user,assistant(JSON)]
    return tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    train_dataset=ds_train, eval_dataset=ds_val,
    formatting_func=fmt, max_seq_length=512,
    args=SFTConfig(
        output_dir=OUT, num_train_epochs=2,
        per_device_train_batch_size=2, gradient_accumulation_steps=16,
        learning_rate=3e-5, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=20, save_steps=500, eval_steps=500,
        fp16=True, gradient_checkpointing=True, report_to="none",
    ),
)
trainer.train()
trainer.model.save_pretrained(OUT); tok.save_pretrained(OUT)

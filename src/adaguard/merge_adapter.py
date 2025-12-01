from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

adapter_repo = "msho/s1_wjb_llama3_peft"
merged_repo = "msho/s1_wjb_llama3_merged_fr"

print("Loading PEFT model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_repo,
    device_map="auto",
    torch_dtype="auto",
)
print("Merging LoRA into base...")
model = model.merge_and_unload()  # returns a plain Transformers model

tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

print("Pushing merged model...")
model.push_to_hub(merged_repo)
tokenizer.push_to_hub(merged_repo)
print("Done.")

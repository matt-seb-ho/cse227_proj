from datasets import load_from_disk

ds = load_from_disk("wildjailbreak_with_gpt_oss_20b/subset_with_gpt_oss_20b")
filtered = ds.filter(lambda x: x["gpt_oss_20b_correct"])

print(ds)
print(filtered)

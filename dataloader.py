from datasets import load_dataset

# Load the full dataset
ds = load_dataset("roskoN/dailydialog", split="train+validation+test")

with open("DailyDialog.txt", "w", encoding="utf-8") as f:
    for ex in ds:
        # Safely access 'dialog' list
        if "dialog" in ex:
            for line in ex["dialog"]:
                f.write(line.strip() + "\n")
            f.write("\n")  # Separate conversations
        else:
            print("⚠️ Warning: Missing 'dialog' in example:", ex)

print("✅ DailyDialog saved as DailyDialog.txt")

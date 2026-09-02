import os
import csv
import random
import yaml

def load_config(config_path="iris_recognition/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def split_dataset():
    config = load_config()
    index_path = config["dataset"]["index_path"]
    splits_dir = config["dataset"]["splits_dir"]
    seed = config["evaluation"]["random_seed"]
    
    print(f"Reading dataset index from: {index_path}")
    if not os.path.exists(index_path):
        print(f"Error: Index file {index_path} does not exist. Run dataset_index.py first.")
        return
        
    # Read rows
    rows = []
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    # Get unique subject IDs
    subjects = sorted(list(set(row["subject_id"] for row in rows)))
    total_subjects = len(subjects)
    print(f"Total unique subjects: {total_subjects}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(subjects)
    
    # Split: 70% Train, 15% Val, 15% Test
    train_size = int(0.70 * total_subjects)
    val_size = int(0.15 * total_subjects)
    # The rest is test
    
    train_subs = sorted(subjects[:train_size])
    val_subs = sorted(subjects[train_size:train_size + val_size])
    test_subs = sorted(subjects[train_size + val_size:])
    
    print(f"Train subjects: {len(train_subs)} ({len(train_subs)/total_subjects:.1%})")
    print(f"Val subjects: {len(val_subs)} ({len(val_subs)/total_subjects:.1%})")
    print(f"Test subjects: {len(test_subs)} ({len(test_subs)/total_subjects:.1%})")
    
    # Write split text files
    os.makedirs(splits_dir, exist_ok=True)
    
    def write_txt(path, sub_list):
        with open(path, "w") as f:
            for s in sub_list:
                f.write(f"{s}\n")
                
    write_txt(os.path.join(splits_dir, "train_subjects.txt"), train_subs)
    write_txt(os.path.join(splits_dir, "val_subjects.txt"), val_subs)
    write_txt(os.path.join(splits_dir, "test_subjects.txt"), test_subs)
    print(f"Splits saved as text files in {splits_dir}")
    
    # Create split maps
    split_map = {}
    for s in train_subs:
        split_map[s] = "train"
    for s in val_subs:
        split_map[s] = "val"
    for s in test_subs:
        split_map[s] = "test"
        
    # Update CSV index with 'split' column
    updated_rows = []
    for row in rows:
        sub_id = row["subject_id"]
        row["split"] = split_map[sub_id]
        updated_rows.append(row)
        
    # Rewrite index CSV with split field included
    with open(index_path, "w", newline="") as csvfile:
        fieldnames = ["subject_id", "eye_side", "image_path", "split", "width", "height", "channels"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
        
    print(f"Dataset index updated with 'split' column at: {index_path}")

if __name__ == "__main__":
    split_dataset()

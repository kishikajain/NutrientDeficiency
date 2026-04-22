import splitfolders

splitfolders.ratio(
    "data/raw_data/dataset",
    output="data",
    seed=1337,
    ratio=(0.8, 0.1, 0.1)
)

print("✅ Data split completed!")
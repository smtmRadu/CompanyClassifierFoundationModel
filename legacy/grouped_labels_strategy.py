import polars as pl







def get_prompt():
    prompt = "Given this list of insurance taxonomy categories, please generate (if possible) a group that contains 5 different categories that are completely unrelated and any company cannot be ever represented by more than one of them (or of course none of them): "
    df = pl.read_csv("labels_groups/labels_remaining.csv")
    big_group = df['label'].to_list()
    print(len(big_group))
    group = pl.read_csv("labels_groups/group37.csv")
    small_group = group['label'].to_list()
    print(len(small_group))

    # remove the group_categories from categories
    for category in small_group:
        if category in big_group:
            big_group.remove(category)

    print(len(big_group))

    df = pl.DataFrame({'label': big_group})
    df.write_csv("labels_groups/labels_remaining.csv")

    with open("labels_groups/grouped_labels_strategy.txt", "w") as f:
        f.write(prompt + ", ".join(big_group))


get_prompt()
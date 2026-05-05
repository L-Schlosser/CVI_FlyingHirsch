from pathlib import Path

# 75-class → 12-class mapping
numbers = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 2,
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 3,
    24: 3,
    25: 3,
    26: 3,
    27: 4,
    28: 4,
    29: 4,
    30: 4,
    31: 4,
    32: 4,
    33: 4,
    34: 5,
    35: 5,
    36: 5,
    37: 5,
    38: 5,
    39: 5,
    40: 5,
    41: 5,
    42: 6,
    43: 6,
    44: 6,
    45: 6,
    46: 6,
    47: 6,
    48: 6,
    49: 7,
    50: 7,
    51: 7,
    52: 7,
    53: 8,
    54: 8,
    55: 8,
    56: 9,
    57: 9,
    58: 9,
    59: 9,
    60: 10,
    61: 10,
    62: 10,
    63: 10,
    64: 10,
    65: 10,
    66: 10,
    67: 10,
    68: 10,
    69: 10,
    70: 11,
    71: 11,
    72: 11,
    73: 11,
    74: 11
}


def convert_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    out_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_class = int(parts[0])

        if old_class not in numbers:
            continue

        new_class = numbers[old_class]

        out_lines.append(
            " ".join([str(new_class)] + parts[1:])
        )

    # OVERWRITE SAME FILE
    with open(file_path, "w") as f:
        f.write("\n".join(out_lines))


def main():
    for split in ["train", "val", "test"]:
        in_dir = Path(f"datasets/labels/{split}")

        for file in in_dir.glob("*.txt"):
            convert_file(file)

    print("Conversion complete (labels overwritten).")


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd


def mbr_to_wkt(min_x, min_y, max_x, max_y):
    """
    Convert an MBR into a closed WKT POLYGON.
    """
    return (
        f"POLYGON(({min_x} {min_y}, "
        f"{max_x} {min_y}, "
        f"{max_x} {max_y}, "
        f"{min_x} {max_y}, "
        f"{min_x} {min_y}))"
    )


def convert_csv_directory_to_wkt(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in input_dir.glob("*.csv"):
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["min_x", "min_y", "max_x", "max_y"]
        )

        wkt_polygons = []

        for _, row in df.iterrows():
            min_x = row["min_x"]
            min_y = row["min_y"]
            max_x = row["max_x"]
            max_y = row["max_y"]

            wkt_polygons.append(mbr_to_wkt(min_x, min_y, max_x, max_y))

        output_path = output_dir / f"{csv_path.stem}.wkt"

        with open(output_path, "w", encoding="utf-8") as f:
            for wkt in wkt_polygons:
                f.write(wkt + "\n")

        print(f"Created: {output_path}")


if __name__ == "__main__":
    INPUT_DIR = "real_datasets_csv"
    OUTPUT_DIR = "real_datasets_wkt"

    convert_csv_directory_to_wkt(INPUT_DIR, OUTPUT_DIR)
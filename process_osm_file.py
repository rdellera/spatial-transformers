from pathlib import Path


def extract_geometry_column_to_wkt(input_csv, output_wkt):
    input_csv = Path(input_csv)
    output_wkt = Path(output_wkt)
    output_wkt.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with open(input_csv, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_wkt, "w", encoding="utf-8") as fout:

        # Skip header
        next(fin, None)

        for line in fin:
            line = line.rstrip("\n\r")

            if not line:
                continue

            # First tab-separated column = geometry
            geometry = line.split("\t", 1)[0].strip()

            if geometry:
                fout.write(geometry + "\n")
                written += 1

    print(f"Created: {output_wkt}")
    print(f"Written geometries: {written}")


def convert_directory(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in input_dir.glob("*.csv"):
        output_wkt = output_dir / f"{csv_path.stem}.wkt"
        extract_geometry_column_to_wkt(csv_path, output_wkt)


if __name__ == "__main__":
    INPUT_DIR = "real_datasets_csv"
    OUTPUT_DIR = "real_datasets_wkt"

    convert_directory(INPUT_DIR, OUTPUT_DIR)
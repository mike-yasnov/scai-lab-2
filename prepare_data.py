import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Распаковка тестовых видео в data/")
    parser.add_argument("--zip", default=str(ROOT.parent / "test-videos.zip"), help="путь к test-videos.zip")
    args = parser.parse_args(argv)
    DATA_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.zip) as archive:
        for member in archive.namelist():
            if not member.endswith(".avi"):
                continue
            target = DATA_DIR / Path(member).name
            target.write_bytes(archive.read(member))
            print(target)


if __name__ == "__main__":
    main()

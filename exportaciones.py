from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def dict_to_json_bytes(data: dict) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def write_dataframe_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=index)
    return target


def write_json(data: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target


def exportar_paquete_resultados(
    output_dir: str | Path,
    series: dict[str, pd.DataFrame],
    resumenes: dict[str, pd.DataFrame],
    metadata: dict,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_path = output / f"resultados_streger_{timestamp}.zip"

    temp_files: list[Path] = []
    for name, df in {**series, **resumenes}.items():
        file_path = output / f"{name}.csv"
        df.to_csv(file_path, index=False)
        temp_files.append(file_path)

    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp_files.append(metadata_path)

    with ZipFile(package_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in temp_files:
            zip_file.write(file_path, arcname=file_path.name)

    return package_path

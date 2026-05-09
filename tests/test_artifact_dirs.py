from pathlib import Path

from infty.analysis.surrogate_quality import plot_summary_records, save_records_as_csv


def test_plot_summary_records_skips_directory_creation_without_records(tmp_path):
    output_dir = tmp_path / "surrogate" / "plots"
    result = plot_summary_records([], output_dir)
    assert result == []
    assert not output_dir.exists()


def test_save_records_as_csv_skips_directory_creation_without_rows(tmp_path):
    output_path = tmp_path / "surrogate" / "summary.csv"
    save_records_as_csv(output_path, [])
    assert not output_path.parent.exists()

import pytest

from main import check_args, parse_args, parse_roi


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    path.write_bytes(b"x")
    return str(path)


def test_parse_roi():
    assert parse_roi(None) is None
    assert parse_roi("select") == "select"
    assert parse_roi("1,2,30,40") == (1, 2, 30, 40)
    with pytest.raises(ValueError):
        parse_roi("1,2,3")
    with pytest.raises(ValueError):
        parse_roi("1,2,0,4")


@pytest.mark.parametrize("extra, fragment", [
    (["--ratio", "1.5"], "Лоу"),
    (["--min-inliers", "3"], "инлаеров"),
    (["--max-coast", "-1"], "max-coast"),
    (["--max-size", "10"], "max-size"),
    (["-o", "out.gif"], ".mp4"),
    (["--label", "Мона"], "латиницей"),
    (["--roi", "a,b,c,d"], "ROI"),
    (["--method", "yolo", "-d", "orb"], "yolo"),
    (["--method", "yolo", "--roi", "0,0,10,10"], "yolo"),
])
def test_rejects_bad_arguments(video, extra, fragment):
    error = check_args(parse_args([video, *extra]))
    assert error is not None and fragment in error


def test_rejects_missing_video():
    assert "не найден" in check_args(parse_args(["nope.avi"]))


def test_accepts_good_arguments(video):
    assert check_args(parse_args([video, "-d", "orb", "--roi", "0,0,10,10", "-o", "out.mp4", "--label", "Mona Lisa"])) is None

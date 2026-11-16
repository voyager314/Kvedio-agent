from pathlib import Path

import yaml

from kvedio_agent1 import main

BINDING = {
    "name": "primary",
    "model": "some-model",
    "api_base": "https://example.invalid/v1",
    "api_key": "",
}


def write_config(home: Path, binding: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    payload = {"roles": {"supervisor": [binding], "asr": [], "video": []}}
    (home / "config.yml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_config_creates_a_template_and_reports_where_it_is(home, capsys):
    assert main(["config"]) == 0
    printed = capsys.readouterr().out
    assert str(home / "config.yml") in printed
    assert "本次按模板创建" in printed


def test_doctor_reports_each_check(home, capsys):
    assert main(["doctor"]) == 0
    printed = capsys.readouterr().out
    for expected in ("[ok] 配置", "[ok] 数据库", "[ok] 解码器"):
        assert expected in printed
    assert "[warn] 监督" in printed


def test_a_literal_key_never_reaches_the_output(home, capsys):
    write_config(home, dict(BINDING, api_key="sk-live-abcdef"))
    assert main(["config"]) == 0
    printed = capsys.readouterr().out
    assert "sk-live-abcdef" not in printed
    assert "已配置" in printed


def test_a_broken_config_is_one_line_and_a_non_zero_status(home, capsys):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yml").write_text("roles: [", encoding="utf-8")
    assert main(["config"]) == 1
    reported = capsys.readouterr().err
    assert "无法启动" in reported
    assert "Traceback" not in reported

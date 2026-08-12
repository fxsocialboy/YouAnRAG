from scripts.answer_stage7 import build_options, parse_args


def test_stage7_cli_exposes_composer_and_judge_options():
    args = parse_args(["台风怎么办", "--composer-mode", "deepseek", "--judge", "--no-mmr"])
    options = build_options(args)

    assert args.judge is True
    assert options.composer_mode == "deepseek"
    assert options.enable_mmr is False

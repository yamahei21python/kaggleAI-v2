"""`python3 -m cli` の入口。

既定（サブコマンド無し）は build のみ。push は明示指定 + `--yes`。

サブコマンドは `set_defaults(func=...)` で自分の実装を持つ。以前は
`main()` が `argv[:1]` を見て quota / ask / timings / beacon を横取りし、
`sys.argv` を書き換えてから委譲していた（サブパーサは `--help` に出すだけで
使われない**死んだ定義**だった）。今は argparse の結果だけを見る。
"""
import argparse
import sys

from cli import build as build_mod
from cli import check as check_mod
from cli import push as push_mod
from kbuild import spec


def _cmd_build(args):
    return build_mod.run(
        args.targets or None,
        dry_run=args.dry_run,
        extra_args=["--smoke"] if args.smoke else None)


def _cmd_push(args):
    return push_mod.push(args.target, yes=args.yes, dry_run=args.dry_run,
                         smoke=args.smoke, no_run=args.no_run)


def _cmd_list(_args):
    print(f"{'status':8s} {'gpu':3s} {'kind':8s} {'id':56s} dir")
    for k in spec.all_kernels():
        print(f"{k.status:8s} {str(k.gpu):3s} {k.kind:8s} {k.id:56s} {k.out}")
    return 0


def _cmd_check(_args):
    return check_mod.run()


# ask / quota / timings / beacon は自前のオプションを持つので、こちらの
# パーサでは中身を見ない（`REMAINDER` で丸ごと受け取って渡す）。重い import
# （requests など）を `cli build` に巻き込まないよう、呼ぶときに読む。
def _cmd_ask(args):
    from cli.ask import main as ask_main
    return ask_main(args.rest)


def _cmd_quota(args):
    from cli.quota import main as quota_main
    return quota_main(args.rest)


def _cmd_timings(args):
    from cli.timings import main as timings_main
    return timings_main(args.paths)


def _cmd_beacon(args):
    from cli.beacon import main as beacon_main
    return beacon_main(args.rest)


def _parser():
    ap = argparse.ArgumentParser(prog="python3 -m cli", description=__doc__)
    # サブコマンド無しは build。`build` の既定値もここで持たせる（下の
    # サブパーサが同じ名前で上書きする）。
    ap.set_defaults(func=_cmd_build, targets=[], dry_run=False, smoke=False)
    sub = ap.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="カーネルを生成する（既定）")
    p_build.add_argument("targets", nargs="*", help="カーネル名（省略時は active 全部）")
    p_build.add_argument("--dry-run", action="store_true")
    p_build.add_argument("--smoke", action="store_true",
                         help="疎通だけの変種を生成（GPU 時間の節約）")
    p_build.set_defaults(func=_cmd_build)

    p_push = sub.add_parser("push", help="生成物を Kaggle へ push する")
    p_push.add_argument("target", help="カーネル名 / id / ディレクトリ")
    p_push.add_argument("--yes", action="store_true", help="実行を確定する")
    p_push.add_argument("--dry-run", action="store_true")
    p_push.add_argument("--smoke", action="store_true",
                        help="smoke 変種を生成して push（トンネル / keepalive 省略）")
    p_push.add_argument("--no-run", dest="no_run", action="store_true",
                        help="保存だけして実行しない（Quick Save。GPU を消費しない。要 kaggle CLI 2.x）")
    p_push.set_defaults(func=_cmd_push)

    sub.add_parser("list", help="台帳を表示する").set_defaults(func=_cmd_list)
    sub.add_parser("check", help="CPU 検証をまとめて実行する").set_defaults(
        func=_cmd_check)

    # ここから下は自前のオプションを持つ。`--help` を含めて**中身を見ない**ので
    # `add_help=False` にして、残りは `parse_known_args` の余りとして受け取る
    # （`REMAINDER` はサブパーサでは先頭の `--opt` を取りこぼす）。
    p_ask = sub.add_parser("ask", help="公開 URL に問い合わせる", add_help=False)
    p_ask.set_defaults(func=_cmd_ask, passthrough=True)

    p_quota = sub.add_parser("quota", help="GPU/TPU クォータ残量", add_help=False)
    p_quota.set_defaults(func=_cmd_quota, passthrough=True)

    p_timings = sub.add_parser(
        "timings", help="result.json から起動時間の内訳を比べる")
    p_timings.add_argument("paths", nargs="*", help="result.json のパス（複数可）")
    p_timings.set_defaults(func=_cmd_timings)

    p_beacon = sub.add_parser(
        "beacon", help="ビーコン（掲示板カーネル）から公開 URL を読む",
        add_help=False)
    p_beacon.set_defaults(func=_cmd_beacon, passthrough=True)
    return ap


def main(argv=None):
    ap = _parser()
    args, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if getattr(args, "passthrough", False):
        args.rest = rest
    elif rest:
        ap.error("unrecognized arguments: " + " ".join(rest))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""`python3 -m cli` の入口。

既定（サブコマンド無し）は build のみ。push は明示指定 + `--yes`。
"""
import argparse
import sys

from cli import build as build_mod
from cli import check as check_mod
from cli import push as push_mod
from kbuild import spec


def _parser():
    ap = argparse.ArgumentParser(prog="python3 -m cli", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="カーネルを生成する（既定）")
    p_build.add_argument("targets", nargs="*", help="カーネル名（省略時は active 全部）")
    p_build.add_argument("--dry-run", action="store_true")
    p_build.add_argument("--smoke", action="store_true",
                         help="疎通だけの変種を生成（GPU 時間の節約）")

    p_push = sub.add_parser("push", help="生成物を Kaggle へ push する")
    p_push.add_argument("target", help="カーネル名 / id / ディレクトリ")
    p_push.add_argument("--yes", action="store_true", help="実行を確定する")
    p_push.add_argument("--dry-run", action="store_true")
    p_push.add_argument("--smoke", action="store_true",
                        help="smoke 変種を生成して push（トンネル / keepalive 省略）")
    p_push.add_argument("--no-run", dest="no_run", action="store_true",
                        help="保存だけして実行しない（Quick Save。GPU を消費しない。要 kaggle CLI 2.x）")

    sub.add_parser("list", help="台帳を表示する")
    sub.add_parser("check", help="CPU 検証をまとめて実行する")

    p_ask = sub.add_parser("ask", help="公開 URL に問い合わせる")
    p_ask.add_argument("rest", nargs=argparse.REMAINDER)

    p_quota = sub.add_parser("quota", help="GPU/TPU クォータ残量")
    p_quota.add_argument("rest", nargs=argparse.REMAINDER)

    p_timings = sub.add_parser(
        "timings", help="result.json から起動時間の内訳を比べる")
    p_timings.add_argument("paths", nargs="*", help="result.json のパス（複数可）")
    return ap


def _list():
    print(f"{'status':8s} {'gpu':3s} {'kind':8s} {'id':56s} dir")
    for k in spec.all_kernels():
        print(f"{k.status:8s} {str(k.gpu):3s} {k.kind:8s} {k.id:56s} {k.out}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # quota / ask は独自のオプションを持つので argparse を通さず委譲する
    if argv[:1] == ["quota"]:
        from cli.quota import main as quota_main
        sys.argv = ["quota"] + argv[1:]
        return quota_main()
    if argv[:1] == ["ask"]:
        from cli.ask import main as ask_main
        sys.argv = ["ask"] + argv[1:]
        return ask_main()
    if argv[:1] == ["timings"]:
        from cli.timings import main as timings_main
        return timings_main(argv[1:])

    args = _parser().parse_args(argv)
    if args.cmd in (None, "build"):
        targets = getattr(args, "targets", None)
        return build_mod.run(
            targets or None,
            dry_run=getattr(args, "dry_run", False),
            extra_args=["--smoke"] if getattr(args, "smoke", False) else None)
    if args.cmd == "push":
        return push_mod.push(args.target, yes=args.yes, dry_run=args.dry_run,
                             smoke=args.smoke, no_run=args.no_run)
    if args.cmd == "list":
        _list()
        return 0
    if args.cmd == "check":
        return check_mod.run()
    return 2


if __name__ == "__main__":
    sys.exit(main())

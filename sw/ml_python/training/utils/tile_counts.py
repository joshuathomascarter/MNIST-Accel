"""tile_counts.py
Helpers to compute tile counts (MT, NT, KT) = ceil(M/Tm), ceil(N/Tn), ceil(K/Tk)

Usage:
    from python.utils.tile_counts import compute_tile_counts
    MT, NT, KT = compute_tile_counts(M, N, K, Tm, Tn, Tk)

Standalone CLI:
    python python/utils/tile_counts.py --M 7 --N 11 --K 22 --Tm 4 --Tn 4 --Tk 8
"""

from __future__ import annotations
import argparse


def ceil_div(a: int, b: int) -> int:
    """Return ceil(a/b). If b==0 returns 0 to avoid ZeroDivisionError.

    Args:
        a: numerator (non-negative int)
        b: denominator (positive int)

    Returns:
        int: ceil(a/b) or 0 if b == 0
    """
    if b == 0:
        return 0
    return (a + b - 1) // b


def compute_tile_counts(M: int, N: int, K: int, Tm: int, Tn: int, Tk: int):
    """Compute ceil-based tile counts for each dimension.

    Args:
        M, N, K: problem dimensions
        Tm, Tn, Tk: tile dimensions

    Returns:
        tuple (MT, NT, KT)
    """
    return (ceil_div(M, Tm), ceil_div(N, Tn), ceil_div(K, Tk))


def _cli():
    parser = argparse.ArgumentParser(description="Compute tile counts MT/NT/KT = ceil(M/Tm) ...")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--Tm", type=int, required=True)
    parser.add_argument("--Tn", type=int, required=True)
    parser.add_argument("--Tk", type=int, required=True)
    args = parser.parse_args()
    MT, NT, KT = compute_tile_counts(args.M, args.N, args.K, args.Tm, args.Tn, args.Tk)
    print(f"MT={MT} NT={NT} KT={KT}")


if __name__ == "__main__":
    _cli()

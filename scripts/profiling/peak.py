"""Measured fp32 matmul peak and GPU spec, run before and after the experiment."""
import argparse

from common import gpu_info, mark_done, measure_peak_tflops, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--unit", choices=("peak_before", "peak_after"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    peak = measure_peak_tflops(iters=5 if args.smoke else 50)
    write_json(f"{args.out_dir}/{args.unit}.json", {"gpu": gpu_info(), "peak": peak})
    mark_done(args.out_dir, args.unit)
    print(args.unit, f"{peak['measured_tflops']:.2f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()

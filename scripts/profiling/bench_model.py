"""E2, E3, E5: interleaved model timing on one fixed 100-frame clip.

E2  1-stream (j) vs 3-stream ensemble predict(), demo path as-is
E3  batch 1/2/4 on stream j, demo inference_similarity as-is
E5  stream j, batch 1, no_grad: eager / eager + CUDAGraph / compile default /
    compile reduce-overhead, plus eager with grad on (demo condition) for reference
"""
import argparse

import torch
import torch.nn.functional as F

from common import (
    FULL_REPEAT,
    SMOKE_REPEAT,
    demo_input_keypoint,
    load_ensemble,
    mark_done,
    raw_module,
    run_interleaved,
    write_json,
)
from protogcn.inference import inference_similarity

J_STREAM = "pt05"


def preprocess(ensemble, index, keypoint):
    """Same steps as CalibratedEnsembleRecognizer._score_one_stream before the model."""
    data = {
        "keypoint": keypoint,
        "total_frames": keypoint.shape[1],
        "label": -1,
        "start_index": 0,
        "modality": "Pose",
        "test_mode": True,
    }
    processed = ensemble.pipelines[index](data)["keypoint"]
    if not isinstance(processed, torch.Tensor):
        processed = torch.FloatTensor(processed)
    return processed.to(ensemble.device)


def similarity_forward(backbone, head, x):
    """inference_similarity with an exchangeable backbone callable."""
    _, z_query = backbone(x)
    f_query = F.normalize(head.csc_loss.cl_fc(z_query), p=2, dim=1)
    m = F.normalize(head.csc_loss.avg_f.T, p=2, dim=1)
    scores = (m @ f_query.T).T / head.csc_loss.tmp
    return F.softmax(scores, dim=1)


def build_cuda_graph(backbone, head, static_x):
    """Capture eager similarity_forward into a torch.cuda.CUDAGraph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            similarity_forward(backbone, head, static_x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_out = similarity_forward(backbone, head, static_x)
    return graph, static_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    repeat = SMOKE_REPEAT if args.smoke else FULL_REPEAT
    _, _, eager_warmup, compiled_warmup = repeat

    keypoint = demo_input_keypoint(args.out_dir)
    ens1 = load_ensemble([J_STREAM])
    ens3 = load_ensemble()
    j_model = ens1.models[0]
    j_raw = raw_module(j_model)
    x1 = preprocess(ens1, 0, keypoint)

    configs = {}
    # E2
    configs["e2_1stream"] = {"fn": lambda: ens1.predict(keypoint), "warmup": eager_warmup}
    configs["e2_3stream"] = {"fn": lambda: ens3.predict(keypoint), "warmup": eager_warmup}
    # E3
    for batch in (1, 2, 4):
        xb = x1.repeat(batch, *([1] * (x1.dim() - 1)))
        configs[f"e3_batch{batch}"] = {
            "fn": (lambda xb=xb: inference_similarity(j_model, xb)),
            "warmup": eager_warmup,
        }
    # E5
    backbone, head = j_raw.backbone, j_raw.cls_head
    static_x = x1.clone()
    graph, static_out = build_cuda_graph(backbone, head, static_x)
    compiled_default = torch.compile(backbone, mode="default")
    compiled_ro = torch.compile(backbone, mode="reduce-overhead")

    def eager_nograd():
        with torch.no_grad():
            return similarity_forward(backbone, head, x1)

    def cuda_graph():
        graph.replay()
        return static_out.clone()

    def compile_default():
        with torch.no_grad():
            return similarity_forward(compiled_default, head, x1)

    def compile_ro():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            return similarity_forward(compiled_ro, head, x1).clone()

    configs["e5_eager_grad"] = {"fn": lambda: similarity_forward(backbone, head, x1), "warmup": eager_warmup}
    configs["e5_eager"] = {"fn": eager_nograd, "warmup": eager_warmup}
    configs["e5_cudagraph"] = {"fn": cuda_graph, "warmup": compiled_warmup}
    configs["e5_compile_default"] = {"fn": compile_default, "warmup": compiled_warmup}
    configs["e5_compile_reduce_overhead"] = {"fn": compile_ro, "warmup": compiled_warmup}

    # Sanity: every E5 variant must agree with eager before timing.
    with torch.no_grad():
        ref = eager_nograd()
        check = {
            name: float((fn() - ref).abs().max())
            for name, fn in (("cudagraph", cuda_graph), ("compile_default", compile_default),
                             ("compile_reduce_overhead", compile_ro))
        }
    print("E5 max abs diff vs eager:", check, flush=True)

    results = run_interleaved(configs, repeat, log=lambda m: print(m, flush=True))
    meta = {
        "repeat": {"sets": repeat[0], "reps": repeat[1], "eager_warmup": eager_warmup,
                   "compiled_warmup": compiled_warmup},
        "input_shape": list(x1.shape),
        "e5_max_abs_diff_vs_eager": check,
    }
    groups = {"e2": "e2_", "e3": "e3_", "e5": "e5_"}
    for unit, prefix in groups.items():
        payload = {"meta": meta, "results": {k: v for k, v in results.items() if k.startswith(prefix)}}
        write_json(f"{args.out_dir}/{unit}.json", payload)
        mark_done(args.out_dir, unit)


if __name__ == "__main__":
    main()

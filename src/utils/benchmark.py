
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity


def benchmark_forward(fn, *inputs, repeats=10, desc="", enable_amp=False,
                      amp_dtype=torch.float16, verbose=False, **kwinputs):
    """Measure runtime of the forward-pass of an arbitrary function
    using Pytorch Benchmark's Timer utility.
    """
    if verbose:
        print(desc, "- Forward Pass:")
    def fn_wrap(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
            fn(*inputs, **kwinputs)
    # warm-up
    for _ in range(repeats):
        fn_wrap(*inputs, **kwinputs)
    t = benchmark.Timer(
        stmt="fn_wrap(*inputs, **kwinputs)",
        globals={"fn_wrap": fn_wrap, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    torch.cuda.empty_cache()
    return m


def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc="", enable_amp=False,
                       amp_dtype=torch.float16, verbose=False, **kwinputs):
    """Measure runtime of a single backward-pass of an arbitrary function
    using PyTorch Benchmark's Timer utility.
    """
    if verbose:
        print(desc, "- Backward Pass:")
    # first do the forward pass
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
    # dummy grad tensor to simulate multiplying with dL/dy
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Gradient shape does not match!")
    # warm-up
    for _ in range(repeats):
        y.backward(grad, retain_graph=True)
    # timer
    t = benchmark.Timer(
        stmt="y.backward(grad, retain_graph=True)",
        globals={"y":y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    torch.cuda.empty_cache()
    return m


# TODO: measured combined time > forward + backward time measured individually, why?
def benchmark_combined(fn, *inputs, grad=None, repeats=10, desc="", enable_amp=False,
                       amp_dtype=torch.float16, verbose=False, **kwinputs):
    """Measure Forward + Backward pass combined of an arbitrary function.
    """
    if verbose:
        print(desc, "- Forward + Backward Pass: ")
    # set up dummy grad tensor
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Gradient shape does not match!")
    # foward + backward
    def f(grad, *inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
            y = fn(*inputs, **kwinputs)
            if isinstance(y, tuple):
                y = y[0]
        y.backward(grad, retain_graph=True)
    # warm-up
    for _ in range(repeats):
        f(grad, *inputs, **kwinputs)
    # timer
    t = benchmark.Timer(
        stmt="f(grad, *inputs, **kwinputs)",
        globals={"f": f, "grad": grad, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    torch.cuda.empty_cache()
    return m


def benchmark_all(fn, *inputs, grad=None, repeats=10, desc="", enable_amp=False,
                  amp_dtype=torch.float16, verbose=False, **kwinputs):
    """Benchmark an arbitrary function: Forward, Backward, combined."""
    return (
        benchmark_forward(fn, *inputs, repeats=repeats, desc=desc, enable_amp=enable_amp,
                          amp_dtype=amp_dtype, verbose=verbose, **kwinputs),
        benchmark_backward(fn, *inputs, grad=grad, repeats=repeats, desc=desc,
                           enable_amp=enable_amp, amp_dtype=amp_dtype, verbose=verbose,
                           **kwinputs),
        benchmark_combined(fn, *inputs, grad=grad, repeats=repeats, desc=desc,
                           enable_amp=enable_amp, amp_dtype=amp_dtype, verbose=verbose,
                           **kwinputs),
    )


def pytorch_profiler(fn, *inputs, desc="", trace_filename=None, enable_amp=False,
                     amp_dtype=torch.float16, verbose=False, backward=True, **kwinputs):
    """Wrap benchmarked functions in Pytorch Profiler for detailed information."""
    if verbose:
        print(desc, "- Pytorch Profiler:")
    # dummy grad tensor
    if backward:
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
        grad = torch.randn_like(y)
    # warm-up
    for _ in range(30):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
            y = fn(*inputs, **kwinputs)
            if isinstance(y, tuple):
                y = y[0]
        if backward:
            # manually clear gradients before backprop
            # e.g. a manual version of optimizer.zero_grad()
            # but here I'm not using an optimizer
            for x in inputs:
                x.grad = None
            y.backward(grad)

    # torch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,        # TODO: how to use this?
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
            y = fn(*inputs, **kwinputs)
            if isinstance(y, tuple):
                y = y[0]
        if backward:
            for x in inputs:
                x.grad = None
            y.backward(grad)

    if verbose:
        print(prof.key_averages().table(row_limit=100))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)

    results = {
        "cuda_memory_usage": sum([evt.cuda_memory_usage for evt in prof.events()]),
        "flops": sum([evt.flops for evt in prof.events()]),
        "cpu_time": sum([evt.cpu_time for evt in prof.events()]),
        "cpu_time_total": sum([evt.cpu_time_total for evt in prof.events()]),
        "cuda_time": sum([evt.cuda_time for evt in prof.events()]),
        "cuda_time_total": sum([evt.cuda_time_total for evt in prof.events()]),
    }

    return results


def benchmark_forward_memory(fn, *inputs, desc="", verbose=False, enable_amp=False,
                          amp_dtype=torch.float16, **kwinputs):
    """Measure CUDA memory footprint of an arbitrary function call.
    """
    if verbose:
        print(desc, "- Forward Pass CUDA Memory Footprint")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()        # mem measure begins
    torch.cuda.synchronize()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
        fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / (2**30)   # TODO: why they used 2**20 * 1000?
    if verbose:
        print(f"Max Forward-pass Memory: {mem:.2f} (GB)")
    torch.cuda.empty_cache()


def benchmark_memory(fn, *inputs, desc="", verbose=False, enable_amp=False,
                          amp_dtype=torch.float16, backward=True, **kwinputs):
    """Measure CUDA memory footprint of an arbitrary function call.
    """
    if verbose:
        print(desc, "- Forward + Backward Pass CUDA Memory Footprint")
    if backward:
        for x in inputs:
            x.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()        # mem measure begins
    torch.cuda.synchronize()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
        y = fn(*inputs, **kwinputs)
        if isinstance(y, tuple):
            y = y[0]
    torch.cuda.synchronize()
    mem_fwd = torch.cuda.max_memory_allocated() / (2**30)   # TODO: why they used 2**20 * 1000?
    if backward:
        grad = torch.randn_like(y)      # TODO: should this grad tensor be excluded?
        y.backward(grad)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / (2**30)
    mem_bwd = mem - mem_fwd
    if verbose:
        print(f"Max Forward-pass Memory: {mem_fwd:.2f} (GB)")
        print(f"Max Backward-pass Memory: {mem_bwd:.2f} (GB)")
        print(f"Max Total Memory: {mem:.2f} (GB)")
    torch.cuda.empty_cache()
    return mem_fwd, mem_bwd, mem
    

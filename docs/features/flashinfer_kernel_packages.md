# FlashInfer Kernel Packages

FlashInfer 0.7.5 or later loads the sealed kernel packages named by
`FLASHINFER_KERNEL_PACKAGES` when it is imported and serves its own APIs from them
([Trace Apply](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/trace_apply.rst)).
A package whose Definition is tagged `fi_api:<module>.<function>` also serves a vLLM
function decorated with `vllm.model_executor.kernel_packages.kernel_site`, such as
`vllm.model_executor.layers.mamba.ops.ssd_chunk_scan._chunk_scan_fwd`. A call runs the
package's Solution, with the Definition's inputs as keyword arguments, when every input
is passed, tensor shapes bind the Definition's axes, const axes and dtypes match, and no
other argument is set; any other call runs the function itself. Outputs the function
writes in place, such as `out`, are Definition inputs.

```bash
export FLASHINFER_KERNEL_PACKAGE_ROOT=/opt/flashinfer/kernel_packages  # default
export FLASHINFER_KERNEL_PACKAGES=<digest>[,<digest>...]
vllm serve <model>
```

The log names each package loaded and each site's first hit and fallback reasons;
`flashinfer.trace_apply.stats()` counts them. `FLASHINFER_KERNEL_PLUGINS_DISABLE=1`
turns packages off. A package with a setup step keeps state prepared from the weights,
so vLLM refuses it together with sleep mode, EPLB, weight transfer, CPU offload or
`reload_weights`.

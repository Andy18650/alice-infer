# Shared-B format comparison

Single gfx90a GCD HIP0/BDF0000:31:00.0; CPU+memory NUMA0, automatic clocks, no model/context.
70B cases are TP4/8 per-rank projection shapes, NOT full-model/collective throughput.
Row-major device FP16 A, finite E4M3FN B, logical FP32 K128/N128 scales.
No A/B repacking or FP32 conversion in either path. B is packed once outside timing.
Two paired seeds/order-reversed repeats; means of nine-sample process medians.
GEMM event covers the complete kernel; decode event covers partial+ordered scale/combine.
GB/s below is logical N*K bytes / complete decode event, NOT measured physical HBM traffic.

## Decode: useful weight GB/s

|Case|M|N|K|native|plain|block128|block256|
|---|---:|---:|---:|---:|---:|---:|---:|
|70b-tp4-qkv-decode|1|2560|8192|377.190|391.263|396.588|336.517|
|70b-tp4-o-decode|1|8192|2048|369.878|399.471|397.943|372.497|
|70b-tp4-gateup-decode|1|14336|8192|678.708|838.862|701.757|674.947|
|70b-tp4-down-decode|1|8192|7168|583.846|707.132|523.569|499.664|
|70b-tp8-qkv-decode|1|1280|8192|206.090|218.456|219.186|214.172|
|70b-tp8-o-decode|1|8192|1024|205.606|235.111|231.475|227.458|
|70b-tp8-gateup-decode|1|7168|8192|587.203|685.344|522.052|497.292|
|70b-tp8-down-decode|1|8192|3584|504.825|542.101|528.825|453.649|

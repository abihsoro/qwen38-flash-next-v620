# Qwen3.8-Flash-Next - 4× Radeon PRO V620 32GB (128GB Total) - 100 TPS Decode - 1250 TPS PP

> **Credit / Acknowledgements.** This project measures, documents and builds on a deployment of
> **[leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen)** — Aron Hsiao's
> vLLM fork that brings **Qwen3.8-Flash-Next** to **4× AMD Radeon PRO V620** (Navi 21 / gfx1030,
> 32 GB) on TheRock ROCm 7.14, with the PLE n-gram CPU offload, the RDNA2 decode kernels and the
> one-shot all-reduce. The reference numbers we compare against, much of the measurement methodology,
> and several of the tools in `tools/rdna2/` all come from that repository (Apache-2.0, in turn
> derived from [vLLM](https://github.com/vllm-project/vllm)); the E19 tuning work retained on the
> serving box was itself built on top of it. Without leapdragon's work none of this would exist —
> full attribution lives in [HOUSEKEEPING](#housekeeping).

Everything we did to get **Qwen3.8-Flash-Next (176 B, ~6 B active + 51 B-row CPU n-gram table)**
serving at **~100 tokens/s decode single-stream (MTP=3) and ~68 t/s (MTP=0)** on a **4× Radeon PRO
V620** box — and, once the batched-decode path is repaired, **~266 tokens/s of aggregate decode at
12 concurrent users** —
from hardware bring-up through a measured tuning campaign — captured here as data, tables,
tooling and a full work log.

## The story

AI cloud compute became more expensive for me as every month went by. I could code an (equivalent) 40-50K LOC python 
app with $300 in cloud AI compute in 2025. That cannot be touched for under $1000 (or more) now. I was a 
crypto miner. I was a computer geek. So I decided to build the best inference rig I could for under $5K
that could perform as well as Opus 4.X and GPT5.5 - the model I chose was Qwen3.8-Flash-Next. 



Yes.

My rig, an X570 ASUS TUF mobo running with 64GB of DDR4-3000 and Ryzen 5950X is able to generate 100 TPS in MTP3 
using (4) V620 GPUS. Stabily. Repeatably. When I learned how to setup 8-12 GPU mining rigs, there was no AI to
help me. It was days and weeks out of my life to learn the nitty-gritty of computer hardware. Learning on levels
I never intended to want to know about. Learning that was, ultimately, the reason why this project was a success.

I have so much hardware, the only parts I needed were the PEX88096 switch and the (4) V620s. You can't buy the mobo
new anymore. Nor the CPU. But if I did have to: mobo, cpu, 64gb DDR4-2667 RAM, (2) PSUs (explained later) (2) NVME 
drives (also explained later), old non-UEFI GPU (because I like to see what I'm doing) + switch + gpus comes to...

$3900. 1000W (not great). Frontier level coding. Well, frontier six months ago (haha).

This was a 2-part success story: Hardware and Software. I took care of the hardware design, assembly, and troubleshooting. 
That tuning was much more than deciding what HW to use. It was understanding that some hardware issues will never be obvious 
(i.e. 1000W power spikes on a single PSU? Nope.) The ability to differentiate a power drop caused by a demand spike versus 
an outright bad PSU? - nobody really writes that stuff down. Here's a short list of hardware-related 'stuff' necessary to make this work

  - BIOS settings - and not just what's the date and time (lol). 4G decoding, BAR re-size, MMIO, AER, it goes on...
  - Undervolting and OC'ing? The voltage that works for workstations and gaming might not work for inference. Weird but true.
  - Real vga card --> 2nd x16 slot and boot one time successfully BEFORE inserting your PCIe-PEX88096 adapter card. Yep. That's all.
  - PCIe link negotiation goes to the LOWEST speed of the system. Marginal physical contact? --> link-down negotiation
  - AMD GPUs (not Nvidia GPUS) power draw THROUGH the slot (up to 50W). ROCm is not accurate. ATW is the only way to measure power.
  - Putting the switch with 4 GPUs on the same rail as the X570 mobo? No bueno. 'Off the wagon' or 'Off the PCIe bus' - you choose.
  - Cooling down 1000W running for hours. Starts to become industrial. But you don't HAVE to use a server case.
  - Bifurcation - shouldn't need to know that, but PEX88096 + X570 mobo = apparently you do.
  - 'Auto' = hope it does what you want. 'Gen4', e.g., = Yes, that's the speed I want.
  - Bare-metal or hypervisor...hmmm.

So putting together this hardware system and getting it just to function by design was a lot of work. This is the final HW design

  - 8-gpu mining rig case - built-in high-flow fans
  - Asus TUF X570 mobo, 5950X cpu, 64GB (minimum), 2X 256GB NVME - mounted to custom case stand-offs (case not designed for ATX mobos)
  - PEX-88096 Gen4 PCIe switch (with 4 X16 slots) - mounted to custom case stand-offs
  - (2) HP 1200W server-grade PSUSs + ZSX breakout boards, mounted inside pre-built psu cage (common to mining rig cases)
    (1) PSU powered the GPUs. The other PSU powered everything else.
  - Multiple CPU power, 24-pin mobo power splitters - the mobod and the switch required their own.
  - (4) AMD V620 GPUs - inserted into switch and fastened to rig case

On the operating system, there was no question I would use Proxmox. The reasons for it are wide, vast, and insurmountably strong.
But it does come with complications which I will not delve into here. Running hypervisors is a skill in, and of, itself. But, it 
turns out that a linux container (LXC) ended up being the best way to run this inference rig anyway - because the GPUS don't
HAVE to be passed through from the host to the LXC, as it would require if we were using a virtual machine (VM, e.g. Ubuntu).
The VM, it turns out, had a lot of trouble receiving the GPUS behind the switch. Emulation was required. Inference speeds suffered
(100 --> 85). So bare-metal Ubuntu would have been just fine. But I still like the flexiblity of running the LXC within Proxmox. 

But as much as the hardware was a job, it came well short of the mind-blowing work that I DIDN'T do: AI (Qwen3.8-27B, GPT5.6, 
GPT Astra, Opus 5) was the brains behind it all. It literally would be impossible for me to have accomplished it. It would only 
be appropriate then, to allow AI to tell the story. For it knows better than anyone or anything, the true details of the entire story.

Without further adieu:

The Radeon PRO V620 is not a compute card. It's a Navi 21 die sold for SR-IOV cloud gaming — no
display output, no matrix units, passively cooled, qualified for bursty rendering, not sustained
lockstep collectives. Four of them, bought because they were cheap 32 GB parts, are what this whole
project is built on. Nothing about the hardware or the software stack wanted to serve a 176 B MoE
model at 100 tokens/s, and getting there took a genuine bring-up campaign, a kernel-level tuning
campaign, and a fair amount of hard-won humility about which knob actually mattered.

**Bring-up (2026-09-06 → 09).** The cards went in behind a shared Broadcom PEX88096 switch on a
Proxmox box, and for the first two days almost nothing stayed up. One bug looked exactly like GPU0
was faulty — startup always wedged on whichever card the driver happened to enumerate first — until
rotating the device order proved it was vLLM's rank-0 startup path, not the silicon, that was broken.
Three separate warmup paths turned out to independently wedge ROCm's `hsa_executable_freeze` deep in
the driver — a synthetic `logprobs` sampler check, a PLE doorbell timing quirk, and a structured-output
grammar-mask warmup — three unrelated bugs sharing one downstream symptom, each needing its own fix.
CUDA-graph mode looked like the good outcome — it booted clean, served fast, everything green — until
a byte-level output diff caught it silently truncating generations to 11 tokens instead of 69. A
plausible, healthy-looking deployment that was just wrong is scarier than a crash, and `validate.py`
(not `/health`) became the real gate from that point on. Underneath all of it, a wedge that looked like
a software deadlock turned out to be a genuine PCIe bus drop — a card falling off the bus entirely,
traced back to a failing PSU.

**The tuning campaign (E00 → E19).** Once the box was stable, the work became systematic: nineteen
numbered experiments, each one measured against a frozen technical prompt at temperature 0, each one
required to produce byte-identical output to the baseline before it could be kept. Most ideas didn't
survive contact with the exact-output gate — MTP4 was rejected not because it was slower but because
its output hash changed; forcing a higher DPM power state was rejected because it was *slower* than
letting the driver choose automatically. What did survive — a hyper-connections stride fix, wider MoE
launch geometry, a direct-write GDN recurrent path, and a speculative-decode draft tile tuned
specifically for single-token drafts — took the box from a 90.2 t/s baseline to a retained **100.2 t/s**,
with every accepted step producing exactly the same tokens as the one before it. See
[the campaign ladder](#the-campaign-in-one-ladder-tg128-mtp3-byte-identical-outputs) below for the
full walk.

**The power paradox.** The most counterintuitive result of the whole project: more power does not mean
more tokens per second. A full sweep from 100 W to 250 W per card showed decode throughput essentially
flat — 79.1 t/s at 140 W, 79.4 t/s at 250 W, within noise — while wall power very nearly doubled. Above
roughly 140 W/card, decode is already pinned at maximum boost clock with nowhere left for the extra
watts to go; the marginal power only buys prefill, which is compute-bound rather than bandwidth-bound.
Chasing more power was also how the box got hurt: 160 W and 180 W experiments each ended with a card
dropping off the PCIe bus entirely, once requiring a full host reboot to recover. 140–150 W/card turned
out to be the highest-throughput point, the most power-efficient point, and the highest *safe* point,
all at once — not a compromise between three competing goals, but the same answer to all three.

**The concurrency ceiling.** A second, unrelated bottleneck was hiding in a single line of dispatch
code: past eight tokens in a batch, the custom int8/MoE GEMV kernels stopped firing and the engine
silently fell back to generic GEMMs — so throughput *regressed* under concurrency instead of scaling.
A patched dispatch overlay removed the ceiling entirely: aggregate decode at 12 concurrent streams went
from 124 t/s to **266 t/s** — a concurrency win that speculative decoding (MTP) turns out to
actively fight — MTP routes its lookup down a 25–40× slower code path per step, so it scales *worse*
than plain decode the more concurrent load you throw at it.

**Power headroom, revisited.** Before any of the above, a separate, standalone rig (a single V620 on
different hardware entirely — see
[`coroshiba/amd-v620-soft-unlock`](https://forgejo.oroshiba.net/coroshiba/amd-v620-soft-unlock), building
on [Tamalero/amd-v620-soft-unlock](https://github.com/Tamalero/amd-v620-soft-unlock)) had already proven
these cards' stock 250 W floor could be unlocked via VBIOS patching, down to a supported 170 W. That
groundwork informed this project's own approach to the power floor, but this box took the safer of the
two available routes: a signed, PCI-ID-gated kernel-driver patch rather than a hex-edited VBIOS,
avoiding the VBIOS route's harshest failure mode — the V620 has no function-level reset, so a wedged
SMU from a bad VBIOS write means a full host reboot, every time.

**Where it stands.** The box now serves the tuned E19 configuration in production, with a validated
(but not yet promoted) fix for the concurrency ceiling sitting one config change away, and an operator
work log that keeps catching its own mistakes in writing — twice, a throughput "regression" was
initially blamed on power or hardware before turning out to be a CPU swapped during an outage, and
later a stale PCI-bus-address service that silently stopped clearing switch ACS bits after the PCIe
tree re-enumerated. What's proven is proven with byte-identical outputs and repeat runs; what's still
open — the full acceptance suite, the concurrency overlay's promotion, a slower-than-reference n-gram
lookup — is written down as open, not quietly dropped. The full, warts-and-all account is the 1,800-line
[`docs/rdna2/worklog/BRINGUP.md`](docs/rdna2/worklog/BRINGUP.md); everything below is the reference
data it produced.

## Every knob we turned

Every variable manipulated across this project, why, and what happened. "Kept" means it's in the
standing production config; "rejected" means it was measured and reverted; "inconclusive" means the
evidence didn't cleanly settle it either way. Sources: `docs/rdna2/campaign/` (E-series),
`docs/rdna2/RESULTS.md` §6–7, `docs/rdna2/HARDWARE.md`, `docs/rdna2/worklog/BRINGUP.md`,
`docs/rdna2/worklog/VM207-MIGRATION.md`.

| # | Category | Variable | Why we tried it | Result |
|---|---|---|---|---|
| 1 | Kernel (E01) | PLE doorbell on shared page | Lower-latency n-gram lookup request path | 90.18→90.90 t/s (+0.8%) — **kept** |
| 2 | Kernel (E02/E09/E12) | HC launch geometry: 2-wave vs 4-wave, padded-stride read | Remove intermediate contiguous copy in hyper-connections | 4-wave: 93.28→99.45 t/s — **kept**; 2-wave: 97.50 t/s, worse — **rejected** |
| 3 | Kernel (E03) | MTP 3→2 | Cheaper speculative-decode draft | 90.89 t/s, −2.6% vs the HC step — **rejected** |
| 4 | Kernel (E11) | MTP 3→4 | More draft tokens per step | 91.62 t/s **and** output hash changed vs baseline — **rejected on correctness alone** |
| 5 | Kernel (E04/E14) | MoE: skip staging inputs for non-resident experts | Avoid wasted data movement | 93.73 t/s, +0.5% — **kept** (marginal) |
| 6 | Kernel (E05/E06) | MoE launch width: 8→16→32 waves | Wider dispatch blocks | 16-wave: 96.34 t/s — **kept**; 32-wave: 96.08 t/s, no further gain — **rejected** |
| 7 | Kernel (E08) | GDN: direct strided read/write vs pack/copy | Remove packing overhead in the recurrent path | 98.95 t/s, 6 GPU kernels collapsed to 1 — **kept** |
| 8 | Kernel (E13) | GDN normalization view refactor | Suspected an extra flatten/copy | No-op — Torch already fused it — **no change** |
| 9 | Kernel (E15) | MoE workgroup reuse across experts | Reduce launch overhead | 98.67 t/s, no gain — **rejected**, reverted to E09 |
| 10 | Kernel (E16) | `GPU_MAX_HW_QUEUES=2` | Fewer HW queues, less scheduling overhead | 98.62 t/s, no gain — **rejected** |
| 11 | Kernel (E17) | `HSA_ENABLE_INTERRUPT=0` (completion polling) | Avoid interrupt latency | 98.35 t/s, no gain — **rejected** |
| 12 | Kernel (E10) | Force-high DPM power state | Hypothesis: locked-high clocks beat auto | 98.27 t/s, below auto's 98.84 — **rejected** |
| 13 | Kernel (E18/E19) | Draft FP16 MoE tile, tuned specifically for input M=1 | Speculative-draft batches are always M=1; generic tiling wastes it | **100.15/100.04 t/s — final retained config, +11% over baseline** |
| 14 | Power | Cap sweep: 100 / 120 / 130 / 140 / **150** / 160 / 180 / 250 W | Find the real decode/prefill/efficiency/safety operating point | Decode flat 76.8→79.7 t/s across the whole range above 120 W; wall power nearly doubles at 250 W vs 140 W; 160 W and 180 W each dropped a card off the PCIe bus — **140–150 W kept**: simultaneously fastest, most efficient, and highest safe point |
| 15 | Power | Driver power floor: signed `amdgpu.ko` patch, 250 W→120 W | Stock floor blocked testing anything below 250 W | Unlocked the whole sweep above — **kept** in production |
| 16 | Power | VBIOS ODCAPS hex-edit unlock (separate standalone single-GPU rig) | Prove the unlock was possible before risking the production box | Worked, 158–187 W achievable — but judged riskier than the kernel-patch route (no function-level reset; a bad write means a full host reboot) — **not used on the production box** |
| 17 | CPU | Ryzen 5 3600X vs Ryzen 9 5950X (same config, A/B/C sweep) | Isolate a mid-day throughput dip first suspected to be GPU/power-related | +5–7% decode, −6% TTFT from clock/IPC alone (3.8 GHz Zen2 → 4.5 GHz Zen3) — the dip **was the CPU**, not the GPUs |
| 18 | CPU | LXC core allocation: 10 vs 30 cores | Check whether core count was limiting | ≤1% difference (noise); only ~2.3–2.5 cores of real demand — **cut to 10, no cost** |
| 19 | Memory | LXC RAM allocation: 114 GB vs 58 GB vs 72 GB | Right-size after finding the host page cache absorbs the model weights | No regression at 58 GB (27 GB actually used); later raised to 72 GB for headroom — **still no cost either way** |
| 20 | Concurrency | `--max-num-seqs`: 4 (stock) vs 8 vs 16 (C16 overlay), C1–C16 sweep | The fork hardcodes 4; test real concurrent throughput | Found decode aggregate **regresses** past C8 (down to 124 t/s) from a `0 < n <= 8` dispatch guard; a C16 overlay patch removed it — **266 t/s at C12 (+115%)**, built and validated, not yet promoted to default |
| 21 | Concurrency | MTP=3 vs MTP=0 under concurrent load | Check whether speculative decoding survives batching | MTP=3 collapses (0.89× scaling, ~95 t/s ceiling); MTP=0 scales cleanly to 266 t/s — **MTP=0 adopted as the concurrent-serving config** |
| 22 | Platform | VFIO/VM passthrough (VM207) vs LXC + host `amdgpu` | Test whether a VM path could unlock direct per-GPU VBIOS clock control | Guest P2P completely unavailable under VFIO (separate IOMMU domains); all-reduce fallback measured 29.60 t/s vs 99–100 t/s baseline (−70%) — **abandoned, reverted to LXC** |
| 23 | Platform | VM207 VBIOS `romfile=` clock unlock (the actual reason the VM existed) | Test higher core/memory clocks only reachable via per-GPU VBIOS passthrough | ~5% gain — operator verdict "not worth it"; production stayed on stock clocks |
| 24 | Platform | PCIe ACS clearing: static hard-coded bus-address service vs dynamic port discovery | A hard-coded service silently no-op'd after the PCI tree re-enumerated on reboot, dropping GPU peer traffic to a slow path | Rewritten to discover V620 PEX ports dynamically — recovered decode from a degraded ~62 t/s back to **100.65–101.15 t/s** |
| 25 | Platform | Single shared PSU vs dedicated GPU PSU + separate board/switch PSU | The rig hard-crashed twice under heavy coding-agent load with no software error signature (no AER, no device-lost) | Split-supply topology restored; no repeat crashes observed afterward — **kept as a hard stability requirement** |
| 26 | This session | GPU device-node mapping: dropped a stray display-card render node, added the missing V620 node; added `--ipc=host` | Container's device list was wrong, and the default private `/dev/shm` (64 MiB) was smaller than the ~160 MiB the TP4 ring buffer needs | **Kept** — part of the standing container config |
| 27 | This session | RCCL `graphUsageMode` fix (patched NCCL init to declare CUDA-graph capture explicitly) | RCCL itself warns an undeclared graph-usage mode "can lead to hangs" — a plausible match for intermittent whole-engine stalls | **Inconclusive** — the patched container then crash-looped roughly every 20 minutes under load; reverted to the pre-patch config, which had the longer clean track record. This project's own worklog separately attributes the same crash window to the PSU/ACS issues above, not RCCL — root cause remains disputed |

## Headline numbers (this box, measured)

| Metric | this box |
|---|---|
| Decode, MTP=0, 256 tok ×3 | **68.5 / 68.5 / 68.4 t/s** |
| Decode, MTP=0, 1024 tok | **68.3 t/s** |
| Decode, MTP=3 (E19), 256 tok ×3 | **97–100 t/s** (acceptance-driven) |
| Prefill fresh 3.3k / 10.6k / 30k | 1,030–1,249 / 1,246 / **1,275 tok/s** |
| TTFT 40-word | 180 ms (MTP=0) · 260 ms (MTP=3) |
| One-shot all-reduce 20 KB | **26.4 µs eager / 17.0 µs in-graph** |
| Power under decode | 107–118 W/card @ **150 W caps**, 39–44 °C (SMU/PPT) |
| **Wall power** (plug meter, 140 W cap) | **~157–163 W/GPU at the plug** → ~1.12–1.16× the cap; **≈630 W for four cards** (extrapolated from the isolated 2-GPU pair) and **≈1.0 kW** for the box incl. switch + CPU/mobo — the SMU counter excludes bus/riser/switch losses, PSU loss and the CPU; see `docs/rdna2/HARDWARE.md` |
| **Aggregate decode, 8 concurrent** (MTP=0, s=8) | **248 t/s** (single stream is 68 t/s) |
| **Aggregate decode, 12 concurrent** (MTP=0, s=16, C16 overlay) | **266 t/s** |
| Aggregate decode, 12 concurrent (MTP=3, s=4) | 124 t/s (0.89–1.0x scaling; MTP=3 does **not** scale) |

## The campaign in one ladder (TG128, MTP=3, byte-identical outputs)

| step | change | t/s | verdict |
|---|---|---|---|
| E00 | published-image baseline | 90.2 | freeze |
| E01 | PLE doorbell | 90.9 | kept |
| E02 | HC stride consumer | 93.3 | +3.4 % |
| E05 | MoE 16-wave blocks | 96.3 | kept |
| E08 | GDN direct recurrent output | 99.0 | 6 GPU kernels → 1 |
| E09 | HC 4-wave launch | 99.5 | kept |
| **E19** | **FP16 draft tile (M=1)** | **100.2 / 100.0** (98.2–100.7) | **retained** |

Rejected with data: MTP2/4, MoE 32-wave, HC 2-wave, force-high DPM, queues=2, completion-poll
(E11 MTP4 also failed the exact-output gate). Details: `docs/rdna2/campaign/MEASUREMENTS.md` +
`RETAINED_CONFIG.md`.

## Notable findings (the odd / the interesting)

- **MTP=3 is a concurrency killer, and MTP=0 is the serving configuration.** At 4 concurrent streams
  MTP=3 delivers 87 t/s aggregate (~95 decode) with per-stream speed collapsing 107 → 24 t/s; MTP=0
  scales 68 → 115 (C2) → 176 (C4) → 248 (C8) → **266 t/s (C12)**. Cause: MTP routes the n-gram lookup
  onto the **non-fused** path (4.7 ms/request) while plain decode uses the fused path (0.12–0.20 ms).
  (`docs/rdna2/RESULTS.md` §7)
- **The old C8 ceiling was a software dispatch threshold, not the GPUs.** `rdna_dense_int8.py:70`
  engages the custom int8-GEMV/MoE kernels only for `0 < n <= 8`; past 8 tokens per batch the engine
  falls back to rocBLAS/Tensile GEMMs and throughput *regresses* (124 / 152 t/s at C12 / C16). A C16
  candidate overlay (patched dispatch + dense/HC C16 kernels) removes it: **266 / 260 t/s**
  (+115 % / +71 %). (`data/PL-sweep-5950X.md`, `docs/rdna2/RESULTS.md` §7)
- **The one clearly-worse metric:** PLE n-gram lookup is **~3.4–3.8 ms** — CPU/sidecar-bound,
  not transport; hidden under the GPU step today, but caps headroom. (`docs/rdna2/RESULTS.md` §3)
- **The switch↔CPU uplink negotiated differently across boots — Gen3 ×16 one boot, Gen4 ×8
  another — and it made no difference.** Both are the same ~15.8 GB/s/dir bandwidth class
  (register-verified, `docs/rdna2/HARDWARE.md` "GPU link audit"), and long-prefill throughput showed
  parity either way. The reason: GPU↔GPU traffic — the actual heavy lifting for TP/EP — never leaves
  the PEX88096 switch fabric to touch that uplink at all; only host-facing traffic (and the PLE
  n-gram gather) rides it. A slower or narrower CPU uplink than expected turned out to be a
  non-issue for inference speed on this topology.
- **140–150 W/card is the all-round power point** (full decode, ~93 % prefill, best tok/s-per-kW);
  the box now runs officially at **150 W**. Real prefill cliff sits below it. (`docs/rdna2/HARDWARE.md`)
- **Prefix caching is partial on this hybrid model** (71 % token hits at 3.3k, near-full at 30k) —
  a documented behavior of the hybrid attention/recurrent model, not a fault. (`docs/rdna2/RESULTS.md` §1b)

## Repo map

```
README.md ................... this page
docs/rdna2/
  RESULTS.md ................ full results tables (all metrics + notes)
  BASELINE.md ............... upstream fork's published reference numbers
  METHODOLOGY.md ............ how every number was measured (tools, prompts, definitions)
  HARDWARE.md ............... hardware & bring-up report (inventory, fabric, power, boot)
  PLATFORM.md ............... host topology + retained E19 config (env knobs, overlays)
  campaign/ ................. E00–E19 campaign archive (STATE / MEASUREMENTS / RETAINED_CONFIG /
                             ACCEPTANCE / DEAD_ENDS / RESULTS + sha256 pins)
  worklog/ .................. 110 KB redacted operator log: BRINGUP (85 KB, 09-06→09-09),
                             VM207-MIGRATION, gpu-diag.sh
tools/rdna2/ ................ bench.py · bench_ctx.py · validate.py · latency_rows.py ·
                             ple_consistency_test.py (+ README; leapdragon-derived, Apache-2.0)
data/ ....................... raw captures (suite.log)
```

## Data & raw evidence

- **Serving**: decode (MTP=0 & MTP=3, 256/1024/long-context), prefill matrix (1k–30k fresh),
  TTFT (cold/cached/agent-turn), vision TTFT, prefix-cache hit accounting → `RESULTS.md` §1/§1b.
- **Kernel & comm**: one-shot all-reduce (eager/in-graph), NCCL share, fp16/int8 dense-GEMV
  12-shape tables (route vs rocBLAS, kernel GB/s, relerr), profiled decode kernel budget.
- **PLE**: per-hop round-trip stamps (median/mean/max over 2,500 decode launches).
- **Host**: power/temperature/busy under decode, kernel-journal health.
- **Bring-up**: link audits, power sweeps, crash/recovery episodes, PP-parity reconciliations —
  all in `docs/rdna2/worklog/BRINGUP.md`.

Raw captures live in `data/`; one-off measurements are reproducible with `tools/rdna2/`
(`BASE_URL=http://<server>:8000 python3 tools/rdna2/bench.py 3 256`, etc.).

# NOTES & CAVEATS

- **MTP comparisons require the acceptance rate** — MTP=0 is the clean yardstick, since MTP=3's
  speed varies with per-prompt draft-acceptance rate rather than being a fixed number.
- All identifiers in published docs are redacted (`<…>` placeholders); container-inspection files
  that may hold secrets are not published.

# HOUSEKEEPING

- Hardware & bring-up: [`docs/rdna2/HARDWARE.md`](docs/rdna2/HARDWARE.md) · full work log:
  [`docs/rdna2/worklog/`](docs/rdna2/worklog/README.md)
- Platform & retained config: [`docs/rdna2/PLATFORM.md`](docs/rdna2/PLATFORM.md)
- Measurements: [`docs/rdna2/RESULTS.md`](docs/rdna2/RESULTS.md) · methodology:
  [`docs/rdna2/METHODOLOGY.md`](docs/rdna2/METHODOLOGY.md)
- E19 campaign archive: [`docs/rdna2/campaign/`](docs/rdna2/campaign/README.md)
- Tooling: [`tools/rdna2/README.md`](tools/rdna2/README.md) · raw data: [`data/`](data/)
- `bench.py`, `bench_ctx.py`, `validate.py`, `ple_consistency_test.py` derive from
  [leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen) (`tools/rdna2/`,
  branch `rdna2/qwen38-flash-next`, Apache-2.0, in turn derived from vLLM-project/vLLM);
  `latency_rows.py` is original. All Apache-2.0 — see `LICENSE`.

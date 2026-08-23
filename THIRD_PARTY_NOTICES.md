# Third-Party Notices

llmproxy is licensed under the Apache License, Version 2.0 (see `LICENSE`).

This file records third-party work that llmproxy adapts, along with the licence
each carries. Entries are listed per llmproxy source file so that the origin of
any adapted component can be traced from the code back to its upstream project.

Where a component is a *concept* rather than adapted source, that is stated
explicitly — ideas are not subject to copyright, and those entries exist as
attribution rather than as a licence obligation.

---

## NVIDIA-NeMo/Switchyard — Apache License 2.0

- Upstream: https://github.com/NVIDIA-NeMo/Switchyard
- Version referenced: 0.2.0
- Licence: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Upstream `NOTICE`, reproduced as required by Apache-2.0 §4(d):

```
Switchyard
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This product is licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License in the LICENSE file at the root of this
repository, or at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
```

(Switchyard's `LICENSE` header carries `Copyright (c) 2024-2026 NVIDIA
CORPORATION & AFFILIATES`. The remainder of its `NOTICE` is a dependency listing
for Switchyard's own third-party packages and does not apply to llmproxy.)

### Adapted source

**`llmproxy/signals.py`** — adapted from Switchyard's tool-signal extraction and
stage scoring (`crates/libsy/src/algorithms/util/tool_signals.rs` and
`crates/libsy/src/algorithms/util/stage.rs`).

Changes made (Apache-2.0 §4(b)):

- Translated from Rust to Python; reimplemented against llmproxy's canonical
  OpenAI-shaped message list rather than Switchyard's internal request type.
- Retained the error-severity tiering (soft / hard / critical), the calibrated
  scoring constants (`SIGNAL_UNIT`, `SCORE_GAIN`, `STALL_MIN_TURN_DEPTH`), the
  `spinning` / `exploring` / `production_intensity` projection, and the
  context-compaction escalation override.
- Extended the error-pattern and tool-name tables; llmproxy's taxonomy covers
  additional client harnesses and is maintained independently of upstream's.
- Dropped Switchyard's LLM-classifier consultation path entirely — llmproxy's
  tier selection is heuristic-only and issues no extra model calls.
- Emits a llmproxy-specific decision-source vocabulary rather than Switchyard's
  `DecisionSource` enum.

### Concepts adopted (attribution, not adapted source)

- Stamping every routing decision with the source that produced it, and
  exporting the scorer's *inputs* alongside its verdict rather than the verdict
  alone (`llmproxy/server.py`, `X-LLMProxy-Route-Reason`).
- Excluding routing policy from transport concerns so it can be unit-tested
  without a network stub.

---

## diegosouzapw/OmniRoute — MIT License

- Upstream: https://github.com/diegosouzapw/OmniRoute
- Licence: MIT

```
MIT License

Copyright (c) 2026 diegosouzapw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Concepts adopted (attribution, not adapted source)

- **`llmproxy/server.py` — health-aware demotion.** The insight that a proxy's
  own stream-lifecycle errors and client disconnects must be excluded from
  provider failure accounting, so that one user cancelling a stream cannot
  cascade into provider cooldowns (`src/shared/utils/circuitBreaker.ts`,
  `isLocalStreamLifecycleError`). Also the practice of demoting a degraded
  candidate by a score multiplier rather than excluding it outright, so an
  exhausted provider ranks last instead of surfacing a misleading 429.
- **`llmproxy/server.py` — prompt-cache affinity.** The application of
  rendezvous hashing to prompt-cache stickiness, and the correctness note that a
  request carrying only a first user turn has no reusable prefix and must be
  excluded from affinity keying, lest distinct conversations collapse onto one
  account (`open-sse/services/combo/promptCacheAffinity.ts`).

llmproxy's implementations of both were written against its own data structures;
no OmniRoute source is reproduced here.

---

## Prior-art algorithms

These predate both projects above and are attributed to their original authors
rather than to any intermediate implementation.

- **Rendezvous (highest-random-weight) hashing** — Thaler, D. G. and
  Ravishankar, C. V., "A Name-Based Mapping Scheme for Rendezvous", University
  of Michigan technical report CSE-TR-316-96, 1996. Used in `llmproxy/server.py`
  for prompt-cache affinity.
- **The power of two random choices** — Mitzenmacher, M., "The Power of Two
  Choices in Randomized Load Balancing", 1996. Informs the shape of the health
  score in `llmproxy/server.py`.

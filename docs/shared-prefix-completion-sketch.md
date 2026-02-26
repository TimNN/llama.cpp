# Shared Prefix Optimization for Multi-Prompt Completions

## Problem

The `/completion` endpoint accepts an array of prompts, but each prompt is
processed independently — even when they share a common prefix. For example:

```json
{
  "prompt": [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is"
  ],
  "n_predict": 32
}
```

Here, the prefix `"The capital of "` (5 tokens) is evaluated three separate
times through the model, once per slot.

## Existing Mechanism: `n_cmpl`

The `n_cmpl` parameter already solves a related problem: generating multiple
completions for a **single** prompt. The flow is:

1. **Task creation** (`server-context.cpp:3005`): When `n_cmpl > 1`, child
   tasks are created via `task.add_child()`. Children share the same tokens but
   get distinct sampling seeds.

2. **Slot assignment** (`server-context.cpp:1694`): The parent task gets a
   slot in `SLOT_STATE_STARTED`. Child tasks get slots in
   `SLOT_STATE_WAIT_OTHER` — they skip prompt processing in the main loop.

3. **KV cache copy** (`server-context.cpp:2681–2700`): After the parent slot
   finishes prompt processing (`SLOT_STATE_DONE_PROMPT`), its KV cache state
   is copied to each child via `slot.copy_state_to(*child)`, which calls
   `llama_memory_seq_cp()` — a cheap metadata operation that shares the
   existing KV cache data under a new sequence ID.

4. **Independent generation**: Parent and children all proceed to
   `SLOT_STATE_GENERATING` and sample independently.

## Proposed Design: Shared Prefix Groups

### Core Idea

At request time, detect the common prefix among multiple prompts. Process that
prefix once in a "leader" slot, copy its KV state to "follower" slots, then
have each follower process only its remaining suffix tokens before generating.

This differs from `n_cmpl` in one key way: with `n_cmpl`, children share the
**entire** prompt and diverge only at generation. With shared prefixes,
followers diverge partway through the prompt — they still need to process their
unique suffix tokens before generation begins.

### Step-by-Step Flow

#### 1. Prefix Detection (HTTP thread, during task creation)

In the completion handler (`server-context.cpp:~2966`), after tokenizing all
prompts into `inputs`, compute the longest common prefix (LCP):

```
// pseudocode
std::vector<server_tokens> inputs = tokenize_input_prompts(...);

if (inputs.size() > 1) {
    size_t lcp_len = compute_common_prefix(inputs);  // pairwise token comparison

    if (lcp_len >= MIN_PREFIX_THRESHOLD) {            // e.g., 32 tokens
        // Create a prefix group (see below)
    }
}
```

The threshold avoids overhead for trivially short shared prefixes where the
copy + suffix processing wouldn't save time over just processing independently.

#### 2. Task Structure: Prefix Groups

Introduce a new relationship between tasks — not parent/child (which implies
identical prompts), but **leader/follower** (which implies shared prefix,
different suffixes).

```cpp
// In server_task (server-task.h):
struct server_task {
    // ... existing fields ...

    // Shared prefix support
    int id_leader = -1;             // -1 if this IS the leader or standalone
    int prefix_len = 0;             // number of shared prefix tokens
    std::vector<server_task> follower_tasks;  // only on the leader

    bool is_leader() const { return !follower_tasks.empty(); }
    bool is_follower() const { return id_leader != -1; }
};
```

Task creation would look like:

```
leader_task.tokens = inputs[0];     // full first prompt
leader_task.prefix_len = lcp_len;

for (size_t i = 1; i < inputs.size(); i++) {
    follower_task.tokens = inputs[i];       // full prompt (prefix + unique suffix)
    follower_task.id_leader = leader_task.id;
    follower_task.prefix_len = lcp_len;
    leader_task.follower_tasks.push_back(std::move(follower_task));
}
```

#### 3. Slot States: New `SLOT_STATE_WAIT_PREFIX`

Add a new slot state (or reuse `SLOT_STATE_WAIT_OTHER`):

```
SLOT_STATE_WAIT_PREFIX  // follower: waiting for leader to finish shared prefix
```

Slot assignment (in `process_single_task`):

- Leader: `SLOT_STATE_STARTED` → processes its full prompt normally.
- Followers: `SLOT_STATE_WAIT_PREFIX` → skip processing until leader reaches
  `prefix_len` tokens.

#### 4. Prefix Completion and KV Copy

After each decode step, check if any leader slot has just finished processing
the prefix portion. This check is analogous to the existing `n_cmpl` check at
line 2681, but triggers at `prefix_len` instead of at the end of the full
prompt:

```cpp
// In the main loop, after successful decode:
for (auto & slot : slots) {
    if (slot.state == SLOT_STATE_PROCESSING_PROMPT && slot.task->is_leader()) {
        // Check if we've processed exactly the shared prefix
        if (slot.n_decoded >= slot.task->prefix_len && !slot.prefix_copied) {
            slot.prefix_copied = true;

            for (auto & follower : slots) {
                if (follower.state == SLOT_STATE_WAIT_PREFIX &&
                    follower.task->id_leader == slot.task->id) {

                    // Copy KV cache state up to prefix_len
                    llama_memory_seq_rm(mem, follower.id, -1, -1);
                    llama_memory_seq_cp(mem, slot.id, follower.id, -1, -1);

                    // Copy timing/position metadata
                    follower.n_decoded = slot.task->prefix_len;
                    follower.prompt = slot.prompt.clone();  // prefix portion

                    // Follower now processes its own suffix
                    follower.state = SLOT_STATE_STARTED;
                    // (or SLOT_STATE_PROCESSING_PROMPT with n_past = prefix_len)
                }
            }
        }
    }
}
```

**Important subtlety**: The follower starts with `n_past = prefix_len` (the
shared prefix is already in its KV cache). It then processes tokens
`[prefix_len .. follower_prompt_len)` — the unique suffix — before entering
`SLOT_STATE_DONE_PROMPT` and beginning generation.

This is the main difference from `n_cmpl`: followers still have prompt
processing work to do after the copy.

#### 5. Interaction with Existing KV Cache Reuse

The slot's existing cache reuse logic (`get_common_prefix` at line 2210)
already handles the case where a slot's previous prompt partially matches the
new one. For the **leader**, this works as-is — it benefits from cache hits
from any prior request.

For **followers**, after receiving the prefix via `llama_memory_seq_cp`, the
slot should set `slot.prompt.tokens` to the prefix tokens so that the normal
`SLOT_STATE_STARTED` path recognizes them as already-cached (`n_past =
prefix_len`) and only processes the remaining suffix tokens.

### Key Design Decisions

#### When to trigger the copy

**Option A: At `prefix_len` boundary (mid-prompt)**
The leader copies to followers as soon as it finishes the prefix portion,
then continues processing its own remaining suffix. Followers start their
suffix processing in parallel in the next batch.

**Option B: After leader finishes full prompt**
Simpler — like `n_cmpl` — but followers must wait for the leader to finish
its entire prompt (including the leader's own suffix), even though the
shared part was done earlier. This wastes time if suffixes are long.

Option A is more efficient but requires careful handling of the copy-point
within the prompt processing loop. Option B is simpler to implement as it
closely mirrors the existing `n_cmpl` mechanism.

**Recommendation**: Start with Option B for simplicity. The main savings come
from avoiding redundant prefix evaluation through the model — even if
followers wait for the leader's full prompt, they skip the expensive prefix
decode. Option A can be a follow-up optimization.

#### Minimum prefix threshold

Too-short prefixes (< 32 tokens) aren't worth the overhead of the copy
operation and extra slot coordination. Make this a parameter (e.g.,
`min_shared_prefix` in the request JSON, defaulting to 32).

#### Interaction with `n_cmpl`

These features should compose. A request like:
```json
{ "prompt": ["prefix A ...", "prefix B ..."], "n_cmpl": 3 }
```
should create a prefix group for the shared prefix, and then each prompt
should spawn `n_cmpl` children after its full prompt is processed. The
hierarchy would be:

```
leader (prompt A) ──► follower (prompt B)
  ├── child A.1         ├── child B.1
  └── child A.2         └── child B.2
```

#### Context shift

Like the existing `n_cmpl` children (line 1978), context shift should be
disabled for slots in a prefix group while the prefix is being shared.
After the copy, each follower operates independently and can use context
shift normally.

### Files to Modify

| File | Changes |
|------|---------|
| `server-task.h` | Add `id_leader`, `prefix_len`, `follower_tasks`, `is_leader()`, `is_follower()` |
| `server-context.cpp` | Prefix detection in completion handler; new slot state; copy logic in main loop; slot assignment for leader/follower groups |
| `server-common.h/cpp` | Add `compute_common_prefix(std::vector<server_tokens>&)` utility |
| `server-queue.cpp` | Handle posting of leader+follower task groups (similar to parent+child) |

### Sequence Diagram

```
Time ──►

Slot 0 (leader):   [===== prefix =====][== suffix A ==][generation A ...]
                                    │
                              copy KV cache
                                    │
Slot 1 (follower):  [wait............][== suffix B ==][generation B ...]
Slot 2 (follower):  [wait................][suffix C][generation C ...]
```

vs. current behavior:

```
Slot 0:  [===== prefix =====][== suffix A ==][generation A ...]
Slot 1:  [===== prefix =====][== suffix B ==][generation B ...]
Slot 2:  [===== prefix =====][== suffix C ==][generation C ...]
```

### Risks and Considerations

- **Memory pressure**: `llama_memory_seq_cp` is cheap (just adds a sequence
  tag to existing KV entries), but during suffix processing, the shared prefix
  KV data must remain alive until all followers have their own diverged state.
  With unified attention caches this is fine; with some memory backends it may
  need verification.

- **Batching efficiency**: While followers wait, their slots are occupied but
  idle. This is the same trade-off as `n_cmpl`. If the server is under high
  load, tying up N slots for one request's prefix group may hurt throughput
  for other concurrent requests.

- **Error handling**: If the leader fails (e.g., context exceeded), all
  followers must also be failed. Mirror the existing `release_slots` pattern
  from `launch_slots_with_parent_task`.

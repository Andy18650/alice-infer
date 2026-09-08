# Session API — Design

Status: living draft. Governs how clients talk to the engine; engine-internal
consequences live in `engine-design.md`.

## 1. Session model

- Every session has an ID assigned explicitly by the server (incrementing
  number, UUID — opaque to the client).
- A session begins with a **new session** call (= engine `reset()`). The
  *full context* is allocated up front; if there is not enough space the
  server must reply with an error.
- This is deliberately inefficient compared to a public-facing server and
  sufficient for the single-user use case. Multi-session may later be bolted
  on; the ID system means the API already supports it.

## 2. Lifecycle

- `new_session(model_id, ttl?, snapshot_slots?)`
  → `{session_id, ttl_granted, snapshot_slots_granted}`
- `converse(session_id, append)` → `{reply, new_ttl}`
- Boilerplate (assumed, not specified here): session status query, TTL
  renegotiation for live sessions.

## API Calls

All API calls are authenticated with the user's API key.

### Call: new_session

Client requests the server to start a new session and get assigned an ID.

- parameters:

  1. model_id: the ID of the model that the client want to converse with.
  2. context_size?: maximum context length requested. Optional, default to infinity (use model maximum)
  3. ttl?: time-to-live guaranteed requested. Optional, default to infinity (use server policy, both client and server are strongly advised to set sane values or this will drain user credit, see billing section for details.)
  4. snapshot_slots?: number of snapshots for conversation rollbacks. Optional, default to 0 (No rollback support)
  5. auxillary, temperature, selector strategy, etc. Optional.


- errors:

  1. model not found
  2. server does not have enough space for the context
  3. invalid user credentials (invalid api key or user does not have access to the specified model)
  4. other boilerplate stuff

- response (success):

  1. session_id: the ID to use for future requests.
  2. context_granted: the granted maximum context size.
  3. ttl_granted: the minimal survival time of the session guaranteed by the server. (user credits will be reserved using this value, see billing section)
  4. snapshot_granted: the granted number of snapshot slots.

- edge cases:
context granted > context requested
ttl granted > ttl requested
snapshot granted > snapshot requested

These "generous server" cases should be noted but they should not create issues.

### Call: get_session_info

Client requests up-to-date information about the session.

- parameters:

  1. session_id: the ID of the session

- errors:

  1. invalid session id. The session is not on this server or user does not own this session.
  2. invalid user credentials (And other boilerplate stuff).

- response:

  1. stashed: (bool) is the current session stashed
  2. ttl: the latest ttl of the session
  3. context_size: the maximum context size of the session
  4. context_used: the amount of context used by the session (can be **different** from pure token accounting due to context compression/eviction)
  5. snapshot_num: the number of snapshots allowed
  6. snapshot_used: the number of snapshots already used by the client
  7. other auxiliary info (model ID, message size limit, etc.)

### Call: set_ttl

Client requests to re-negotiate the TTL of the session. A number of 0 means that the session can be dropped immediately. 
(There is no "drop session" command in current design.)

- parameters:

  1. session_id: the ID of the session
  2. new_ttl: the new TTL requested by the client (required)

- errors:

Same as get_session_info

- response:

  1. ttl_granted: the new granted TTL of the session.

- edge cases:

Same as new_session. The server should not return more TTL than requested, because the user is billed using this value.
When the client requests more than original TTL, the server should in principle never use this as a chance to lower the TTL. The returned value should always be larger or equal to the original.
When the client requests less than original TTL, the server should in principle always grant the amount requested. Although this should not be relied upon by the client for robustness of the API. The client should always use the returned value from the server.

### Call: set_context

Client requests to re-negotiate the maximun context length of the session.

All the same as set_ttl, except that setting the context length smaller than the already used context size results in an error.

### Call: generate_response

Used to send all kinds of information to the server (user prompt, tool result, etc.).
Only new information should be sent to the server. **Do not** re-send the entire conversation history.
There is a token limit for each message. **Do not** send massive files or text blocks.

- parameters:

  1. session_id: the ID of the session
  2. content: the content of the information to the sent to the model. May contain multiple parts. The message may contain the model's output (role "assistant"). The message may contain snapshot markers on round boundary (before or after a single message)
  3. max_tokens(optional): maximum number of tokens that the agent can generate
  4. thinking(optional): the thinking effort to generate the response

- errors:

  1. invalid session id. The session is not on this server or user does not own this session.
  2. invalid user credentials.
  3. message too long.
  4. other text generation errors (content control, etc.)

### Call: stash_session

Client requests to put a session in stash 

## 3. TTL semantics

- A request may carry a Time To Live: the minimum time the session is
  guaranteed to survive after its last activity.
- Granted TTL = `min(ttl_client_request, ttl_server_policy)`; if neither is
  specified, TTL defaults to infinity.
- After a session dies, the ID is invalid and all requests with it are
  rejected. To continue, the client obtains a new ID and re-prefills the
  entire history.
- Every conversation response carries a new TTL value — usually the agreed
  maximum, but it may shrink (e.g., the server bills session storage and the
  user is running out of credits).

## 4. Conversations

- All conversation is by session ID; the client sends only the new part to
  append. The server holds the accumulated context.

## 5. Snapshots (limited rollback, no fork)

- The client requests a number of snapshot slots at session creation; the
  server replies with the number actually assigned. Default: 0 (no snapshots
  kept, no rollback possible).
- Snapshots may be inserted only at message boundaries — e.g., before a tool
  result (to allow cutting the tool result out later) or right after the
  user's prompt (to allow cutting out the thinking block). A point not
  snapshotted in advance cannot be rolled back to.
- Slots are managed as a ring buffer: the latest snapshots are kept, earlier
  ones evicted. The client may also voluntarily drop snapshots, oldest first
  (see stashing).
- Design intent: explicitly SSM-friendly. A state snapshot is the only
  rollback mechanism uniform across KV-cache and recurrent-state layers,
  since SSM state cannot be truncated.

## 6. Stashing

A session can be stashed explicitly by the client or implicitly by the
server. A stashed session lives on slower but larger media (system RAM,
flash storage).

- **Explicit stash** — the client may choose how many snapshots to keep
  (default: all). TTL is re-negotiated: if the client specifies no TTL, the
  *old TTL remains* (waiving the negotiation); the client must explicitly
  request infinity to get the maximum stashed TTL. Granted TTL =
  `min(ttl_client_request, ttl_server_stash_policy)`.
- **Implicit stash** (server-initiated: VRAM pressure, long inactivity) — no
  negotiation; the server may only *extend* the TTL, since TTL is a minimum
  guarantee.
- **Unstash** — explicit call (TTL re-negotiated) or implicit via a
  conversation request (the implied request is the original TTL). This keeps
  TTL sane across inactivity → implicit stash → client request → restore
  cycles.

## 7. Future extensions (anticipated, not specified)

- **True fork from a snapshot** — returns a new session ID; does the full
  allocation dance for the forked context.
- **Client stash** — the server returns the entire state as a blob for the
  client to archive. Combined with negotiating TTL down to 0 (server may drop
  immediately), this stops session billing while preserving continuity.
  Guarantee: the same blob creates a new session with *the exact state*, as
  long as *the same API for the same model* is still open. The blob may be
  compressed and/or encrypted.

## 8. Acknowledged limitations

- **All guarantees are scoped to the server process lifetime.** TTL, stash,
  and snapshot survival protect against resource pressure and time, not
  against server crash or restart; there is no durable session store and no
  reload-after-restart path. (Client stash is the intended escape hatch, and
  even it requires the same API and model to remain open.)
- **Orphaned sessions**: the session ID is held only by the client; with no
  enumeration endpoint, a lost ID plus an infinite TTL leaves a session that
  is unreachable and unreclaimable. (Candidate fix: an administrative
  list/kill endpoint.)

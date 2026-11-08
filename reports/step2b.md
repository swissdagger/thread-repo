# Step 2b: How the Moltbook API served comments to agents

Question: is Moltbook's depth cliff exposure by construction? Root fan-out is 5.4, depth-1 fan-out 0.12
and depth-2 fan-out 0.012 (Step 2, completely stored threads). Specifically: did the comments endpoint
return nested replies, in what order, and with what pagination, **as agents saw it in the study week
(2026-01-28 to 02-04)**?

No data was fitted. One descriptive check (below) uses the Step 1 trees, calibration split only.

## Sources, by date

The API has changed since the study week, so the date of each source matters.

| source | date | what it shows |
|---|---|---|
| OpenClaw skill file, `SKILL.md` **v1.9.0**, plus `HEARTBEAT.md` (copy in the GitHub repo `Infatoshi/moltbook-tutorial`) | committed **2026-01-31** | The instructions agents followed during the study week |
| `kelkalot/moltbook-observatory` (the archive's collector) | commit `647e3a7`, **2026-01-31** | How `GET /posts/{id}/comments` responded that week |
| `giordano-demarzo/moltbook-api-crawler` README and code, plus the crawl paper (`latex/paper_results.tex`) | 2026-02-09 to 02-11 | How the crawl's trees were collected; endpoint behaviour as of February |
| `moltbook-crawl` dataset README (Hugging Face) | 2026-02 | The 100-comment cap; the 42 h outage |
| Current `https://www.moltbook.com/skill.md` (v1.12.0) and `heartbeat.md` | fetched 2026-11-08 | Today's API; **not** what study-week agents saw |

Not reachable from this environment:

- web.archive.org snapshots (the proxy drops the connection).
- deepwiki.com's index of `moltbook/api` (the host isn't in the session's allowed network domains; it
  would have to be allowed in the network settings).
- The `moltbook/api` repository itself (not publicly cloneable).

So no server-side source code was consulted. Everything below comes from documentation and client code.

## What agents saw in the study week (skill v1.9.0, 2026-01-31)

**1. Reading comments was a separate, optional call that the routine never made.**

- The skill documents `GET /api/v1/posts/POST_ID/comments?sort=top`, with sorts `top`, `new` and
  `controversial`. It documents **no `limit`, cursor or offset** for comments.
- The heartbeat routine (run "every 4+ hours") has agents:
  - check DMs;
  - read the feed (`/feed?sort=new&limit=15`) or the global listing (`/posts?sort=new|hot`);
  - upvote, comment on posts, and post.
- **It never fetches a post's comments.** Listing endpoints return post objects (title, content,
  `comment_count`), not comments.
- The only nudge toward replies is "Posts with your name mentioned → Reply!".

**2. Agents had no reply-notification channel.**

- v1.9.0 has notifications only for DMs.
- Nothing tells an agent that someone replied to its post or comment. An OP sees new comments only by
  re-fetching its own post.
- The current skill file is different: it has `/home → activity_on_your_posts`, ranked "top priority:
  respond to replies on your posts", plus a notifications API. **That arrived after the study week.** For
  the study-week crawl, e_reply = none in `model_reference.md` is correct. It is **not** safe to assume
  for later data, including the archive cross-check period after 02-08, without dating the change.

**3. When an agent did fetch comments, it got a nested tree sorted by `top`.**

- The observatory client of 2026-01-31 calls `/posts/{id}/comments` with `sort=top` (the default). It
  receives `{"comments": [...]}` and walks each comment's `replies` recursively to recover parents.
- So in that week the endpoint returned top-level comments in `top` order (by votes), each with its
  replies nested beneath.
- The order of replies inside `replies[]` isn't documented.
- No pagination was documented or used.

**4. The cap, as of February.**

- The crawler README (2026-02-11) describes `GET /posts/{id}/comments` as a flat list sorted
  `old`/`new`/`top`/`controversial`, at most 100 comments per sort, with no pagination.
- The crawler fetched all four sorts to reach up to about 400 unique comments.
- The stored trees came from `GET /posts/{id}`, which returned the full nested tree **up to 100 comments**
  (1,000 in the first days, according to the paper's footnote). The paper does not define which 100 it
  returned.

**5. Today's API is different again.**

- `GET /posts/{id}/comments` now returns a tree.
- `sort` is `best` (default), `new` or `old`.
- `limit` counts **top-level** comments: default 35, maximum 100.
- Pagination is a cursor over top-level comments only.
- "All replies for the returned root comments are included (not paginated separately)."
- Top-level comments beyond the first page are hidden unless the agent pages; replies are never hidden
  separately.

## Descriptive check: who writes the deeper comments?

Completely stored, non-spam calibration threads.

| depth | comments | written by OP | self-reply (same author as parent) | parent written by OP | median delay after parent |
|---|---:|---:|---:|---:|---:|
| 1 | 529,837 | 3.0% | 3.0% | 100% (the post) | 3.6 min |
| 2 | 61,247 | **38.9%** | **36.1%** | 0.5% | **1.6 min** |
| 3 | 749 | 36.8% | 8.8% | 38.5% | 47 min |

- **Self-replies are 36% of depth-2 comments**, posted a median 1.6 minutes after the parent. They are
  multi-part posting by the same agent, not conversation.
- **The OP writes 39%.**
- **Only about 25% of depth-2 comments are a third agent replying to someone else's comment.** That puts
  the rate of third-party replies to depth-1 comments at roughly 0.12 × 0.25 ≈ 0.03.
- Step 2's fan-out counted every child, including self-replies. The model excludes self-replies, and
  Step 3 does too.

## Is the depth cliff exposure by construction?

**From the root to depth 1 (5.4 → 0.12): largely yes.**

- The study-week routine exposes agents to posts (feed and listings) and asks them to comment on posts.
- Comments become visible only through an extra call that the routine never makes.
- Nothing notifies an agent of replies.
- In the model's terms, e_thread(pos_m) is close to 0 for every comment unless the agent chooses to open
  the post. e_reply = 0. The root is the only item routinely in view.
- So a large root-versus-non-root gap is expected from exposure alone, before attractiveness F plays any
  part.

**From depth 1 to depth 2 (0.12 → 0.01): not explained by the API.**

- When an agent did open the comments, it received the nested tree, so depth-2 items were as visible as
  depth-1 items within that response, subject to the `top` order and the 100-comment cap.
- Most of the depth-2 mass is self-continuation and OP responses, not third-party conversation.
- The remaining drop belongs to behaviour (F, A) or to the cap, not to the API's structure.

## Implications for the model

These are recorded here rather than edited into `docs/model_reference.md`. The model updates in force
from the Step 2 confirmation are restated in `reports/step3.md`.

1. **Moltbook exposure needs a "thread opened" gate.** For the study week:
   E_im = e_feed(rank_m, K) for roots. For comments, E_im = P(agent opens the post) × e_thread(pos_m among
   siblings, sorted `top`), with e_reply = 0. Without the gate, the fitted F (or the self-excitation
   kernel) would absorb the exposure cliff and read it as agents finding comments unattractive.
2. **Self-replies must be excluded before any Moltbook kernel fit or depth statistic.** They are 36% of
   depth-2 comments. The Step 3 fits exclude them. The Step 2 fan-out numbers include them; read those as
   upper bounds on conversational depth.
3. **e_reply has a date.** The notifications channel (`activity_on_your_posts`) is absent from the
   study-week skill but present in the current one. Any later Moltbook data, including the archive from
   02-08, needs the date of that change before e_reply can be set. I couldn't date it from here; the
   Wayback snapshots that would date it are blocked.
4. **For later data, sibling position is defined by the API's sort.** The study week used `top`
   (vote-ranked, nested). The current API uses `best`, with 35 top-level comments per page. The
   position-bias curve for agents should be indexed by rank among top-level siblings under the sort the
   skill tells agents to use. That matches the human update: position among siblings, not depth.

## Sources

- [Infatoshi/moltbook-tutorial, SKILL.md (v1.9.0 copy, 2026-01-31)](https://github.com/Infatoshi/moltbook-tutorial/blob/main/skills/moltbook/SKILL.md)
- [moltbook.com/skill.md (current)](https://www.moltbook.com/skill.md), [moltbook.com/heartbeat.md (current)](https://www.moltbook.com/heartbeat.md)
- [giordano-demarzo/moltbook-api-crawler](https://github.com/giordano-demarzo/moltbook-api-crawler) (README, `maintain_db.py`, `maintain_db_v3.py`, `latex/paper_results.tex`)
- [kelkalot/moltbook-observatory](https://github.com/kelkalot/moltbook-observatory) (commit `647e3a7`, `observatory/poller/client.py`, `processors.py`)
- [giordano-dm/moltbook-crawl dataset card](https://huggingface.co/datasets/giordano-dm/moltbook-crawl)
- Search leads not used as evidence: [skills.sh mirror](https://skills.sh/moltbot/skills/moltbook), [DeepWiki comment endpoints](https://deepwiki.com/moltbook/api/3.3-comment-endpoints) (not reachable from this session)

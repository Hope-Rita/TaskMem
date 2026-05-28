"""Phase-Two episodic data preparation.

Two subcommands:

    * ``pairs``      build DPO preference pairs from per-video rollout
                     judgements. For each video, build the win-over-lose DAG
                     across all rollout judgements, find the undefeated
                     "winner" with the longest preference chain leading to
                     it, and emit one (winner, loser) pair subject to a
                     token-length-balance filter.
    * ``validation`` sample a held-out set of per-clip episodic memories
                     used for offline evaluation of the steer adapter.
"""
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Graph helpers                                                               #
# --------------------------------------------------------------------------- #


class Node:
    def __init__(self, id: int, value: dict):
        self.id = id
        self.value = value
        self.neighbors: List["Node"] = []

    def add_neighbor(self, neighbor: "Node") -> None:
        self.neighbors.append(neighbor)


class Graph:
    def __init__(self, nodes: Optional[List[Node]] = None):
        self.nodes: List[Node] = nodes if nodes is not None else []

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def _has_cycle_dfs(self, node: Node, visited: Set[Node], stack: Set[Node]) -> bool:
        visited.add(node)
        stack.add(node)
        for nb in node.neighbors:
            if nb not in visited:
                if self._has_cycle_dfs(nb, visited, stack):
                    return True
            elif nb in stack:
                return True
        stack.remove(node)
        return False

    def has_cycle(self) -> bool:
        visited: Set[Node] = set()
        stack: Set[Node] = set()
        return any(
            node not in visited and self._has_cycle_dfs(node, visited, stack)
            for node in self.nodes
        )

    def topological_sort(self) -> List[Node]:
        visited: Set[Node] = set()
        order: List[Node] = []

        def dfs(node: Node) -> None:
            visited.add(node)
            for nb in node.neighbors:
                if nb not in visited:
                    dfs(nb)
            order.append(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)
        return order[::-1]

    def sinks(self) -> List[Node]:
        return [node for node in self.nodes if not node.neighbors]

    def longest_distance_to_sink(self, sink: Node,
                                 topo_order: Optional[List[Node]] = None) -> Dict[Node, int]:
        """Longest path length from every node to `sink` in the DAG; -1 if unreachable."""
        if topo_order is None:
            topo_order = self.topological_sort()
        dis: Dict[Node, int] = {node: -1 for node in self.nodes}
        dis[sink] = 0
        for node in reversed(topo_order):
            if node is sink:
                continue
            best = -1
            for nb in node.neighbors:
                if dis[nb] != -1:
                    best = max(best, dis[nb] + 1)
            dis[node] = best
        return dis


# --------------------------------------------------------------------------- #
# Selection logic                                                             #
# --------------------------------------------------------------------------- #


def _extract_description(response: str) -> str:
    """Best-effort extraction of the actual ``description`` field from a model
    response, falling back to the raw post-``</think>`` tail on parse failure."""
    tail = response.split("</think>")[-1].strip()
    if tail.startswith("```json"):
        tail = tail[len("```json"):].strip()
    if tail.endswith("```"):
        tail = tail[:-3].strip()
    try:
        obj = json.loads(tail)
        if isinstance(obj, dict) and "description" in obj:
            return obj["description"]
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return tail


def _node_token_len(node: Node, tokenizer) -> int:
    if "_sel_len" not in node.value:
        text = _extract_description(node.value["response"])
        node.value["_sel_len"] = len(tokenizer.encode(text, add_special_tokens=False))
    return node.value["_sel_len"]


def _is_negative(node: Node) -> bool:
    return node.value.get("correctness", 0) == 0


def _select_best_pair(graph: Graph, tokenizer,
                      len_threshold: int) -> Tuple[Optional[Tuple[Node, Node]], int, int]:
    """For each undefeated sink x, find the y with the longest preference
    chain ending at x subject to:

        * |len(x) - len(y)| < ``len_threshold``     (token-balance)
        * not both x and y are negative              (drop noisy neg/neg pairs)

    Across all sinks, pick the (x, y) with the longest chain; break ties by
    smaller token-length difference.

    Returns ``((winner, loser), chain_length, length_diff)`` or
    ``(None, -1, +inf)`` if no admissible pair exists.
    """
    topo_order = graph.topological_sort()
    best_pair: Optional[Tuple[Node, Node]] = None
    best_dis = -1
    best_delta = float("inf")

    for x in graph.sinks():
        dis = graph.longest_distance_to_sink(x, topo_order=topo_order)
        len_x = _node_token_len(x, tokenizer)
        candidates = []
        for y in graph.nodes:
            if y is x or dis[y] < 0:
                continue
            delta = abs(len_x - _node_token_len(y, tokenizer))
            if delta >= len_threshold:
                continue
            if _is_negative(x) and _is_negative(y):
                continue
            candidates.append((dis[y], delta, y))
        if not candidates:
            continue
        candidates.sort(key=lambda t: (-t[0], t[1], t[2].id))
        cur_dis, cur_delta, y = candidates[0]
        if cur_dis > best_dis or (cur_dis == best_dis and cur_delta < best_delta):
            best_dis, best_delta, best_pair = cur_dis, cur_delta, (x, y)

    return best_pair, best_dis, (best_delta if best_pair is not None else float("inf"))


# --------------------------------------------------------------------------- #
# Subcommand: pairs                                                           #
# --------------------------------------------------------------------------- #


MEM_TYPE = "episodic"


def _load_video_pairs(args: argparse.Namespace) -> Dict[str, List[list]]:
    """Walk ``train_{split}.jsonl`` files and group the rollout-judgement
    pairs by ``video_id``."""
    pairs_filename = args.pairs_template.format(q_type=args.q_type)
    by_video: Dict[str, List[list]] = defaultdict(list)
    for split in args.splits:
        info_path = os.path.join(args.video_info_dir, f"train_{split}.jsonl")
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            for line in f:
                meta = json.loads(line)
                pairs_path = os.path.join(meta["memory_path"], pairs_filename)
                if not os.path.exists(pairs_path):
                    continue
                try:
                    with open(pairs_path) as f2:
                        for pair_line in f2:
                            pair = json.loads(pair_line)
                            assert pair[0]["reward"] == 1
                            if pair[0]["correctness"] >= pair[1]["correctness"]:
                                video_id = pair[0]["input"][1]["video"].split("/")[-1][:-4]
                                pair[0]["type"] = pair[1]["type"] = MEM_TYPE
                                by_video[video_id].append(pair)
                except Exception as exc:
                    print(f"[warn] failed to read {pairs_path}: {exc}")
    return by_video


def _report_stats(output_path: str, tokenizer) -> None:
    pos_pos = pos_neg = neg_neg = err = 0
    desc_char_deltas: List[int] = []
    raw_char_deltas: List[int] = []
    winner_token_lens: List[int] = []
    loser_token_lens: List[int] = []
    with open(output_path) as f:
        for line in f:
            pair = json.loads(line)
            try:
                desc_0 = _extract_description(pair[0]["response"])
                desc_1 = _extract_description(pair[1]["response"])
            except Exception:
                err += 1
                continue
            desc_char_deltas.append(len(desc_0) - len(desc_1))
            raw_char_deltas.append(len(pair[0]["response"]) - len(pair[1]["response"]))
            winner_token_lens.append(len(tokenizer.encode(desc_0, add_special_tokens=False)))
            loser_token_lens.append(len(tokenizer.encode(desc_1, add_special_tokens=False)))
            c0, c1 = pair[0].get("correctness"), pair[1].get("correctness")
            if c0 == 1 and c1 == 1:
                pos_pos += 1
            elif c0 == 1 and c1 == 0:
                pos_neg += 1
            elif c0 == 0 and c1 == 0:
                neg_neg += 1
            else:
                err += 1
    total = pos_pos + pos_neg + neg_neg
    if total == 0:
        print(f"[stats] no admissible pairs in {output_path}")
        return
    print(f"[stats] total={total} pos/pos={pos_pos} ({pos_pos / total:.4f}) "
          f"pos/neg={pos_neg} ({pos_neg / total:.4f}) "
          f"neg/neg={neg_neg} ({neg_neg / total:.4f}) errors={err}")
    print(f"[stats] avg desc char delta:  {sum(desc_char_deltas) / len(desc_char_deltas):.2f}")
    print(f"[stats] avg raw  char delta:  {sum(raw_char_deltas) / len(raw_char_deltas):.2f}")
    print(f"[stats] avg winner tokens:    {sum(winner_token_lens) / len(winner_token_lens):.2f}")
    print(f"[stats] avg loser  tokens:    {sum(loser_token_lens) / len(loser_token_lens):.2f}")


def build_pairs(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    by_video = _load_video_pairs(args)

    selected: List[list] = []
    for video_id, pairs in tqdm(by_video.items(), desc="building pairs"):
        response_to_node: Dict[str, Node] = {}
        graph = Graph()
        next_id = 0
        for pair in pairs:
            for entry in pair:
                if entry["response"] not in response_to_node:
                    node = Node(next_id, entry)
                    next_id += 1
                    response_to_node[entry["response"]] = node
                    graph.add_node(node)
        for pair in pairs:
            worse = response_to_node[pair[1]["response"]]
            better = response_to_node[pair[0]["response"]]
            worse.add_neighbor(better)

        if graph.has_cycle():
            continue

        best_pair, _, _ = _select_best_pair(graph, tokenizer, args.len_threshold)
        if best_pair is None:
            continue
        winner, loser = best_pair
        winner.value["id"] = loser.value["id"] = video_id
        winner.value["reward"] = 1
        loser.value["reward"] = 0
        selected.append([winner.value, loser.value])

    rng = random.Random(args.seed)
    rng.shuffle(selected)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for pair in selected:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[pairs] wrote {len(selected)} pairs to {args.output_path}")

    if args.report:
        _report_stats(args.output_path, tokenizer)


# --------------------------------------------------------------------------- #
# Subcommand: validation                                                      #
# --------------------------------------------------------------------------- #


def build_validation(args: argparse.Namespace) -> None:
    """Sample a fixed number of episodic-memory records as a held-out set.

    Walks every video in ``video_info_path`` and every clip in
    ``[start_time, end_time)`` at ``clip_seconds`` granularity, loads the
    pre-computed ``{id}_{clip_idx}_{memory_tag}.json`` file under
    ``memory_root/{id}/{clip_idx}/``, shuffles the resulting records, and
    keeps the first ``sample_size`` of them.
    """
    mems: List[dict] = []
    with open(args.video_info_path) as f:
        for line in f:
            data = json.loads(line)
            for t in range(data["start_time"], data["end_time"], args.clip_seconds):
                clip_idx = int(t // args.clip_seconds)
                memory_path = os.path.join(
                    args.memory_root,
                    data["id"],
                    str(clip_idx),
                    f"{data['id']}_{clip_idx}_{args.memory_tag}.json",
                )
                memory_io = json.load(open(memory_path))
                mems.append({
                    "id": f"{data['id']}*{clip_idx}",
                    "type": MEM_TYPE,
                    "input": memory_io[MEM_TYPE]["input"],
                    "output": memory_io[MEM_TYPE]["output"],
                })

    random.seed(args.seed)
    random.shuffle(mems)
    selected = mems[: args.sample_size]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for res in selected:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"[validation] wrote {len(selected)} records to {args.output_path}")


# --------------------------------------------------------------------------- #
# Arg parsing                                                                 #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)

    p = subparsers.add_parser("pairs", help="build DPO preference pairs")
    p.add_argument("--q_type", required=True,
                   help="task type used in the rollout filename "
                        "(default template: S2_{q_type}_pairs.jsonl)")
    p.add_argument("--video_info_dir", required=True,
                   help="directory containing train_{split}.jsonl files; "
                        "each line needs a memory_path field pointing at the "
                        "folder src/main.py --generate_episodic wrote")
    p.add_argument("--splits", nargs="+", type=int, default=[3, 4],
                   help="which train_{split}.jsonl files to scan")
    p.add_argument("--output_path", required=True,
                   help="destination .jsonl of (winner, loser) pairs")
    p.add_argument("--pairs_template", default="S2_{q_type}_pairs.jsonl",
                   help="filename under each video's memory_path that holds "
                        "the rollout-judgement pairs (a python format string "
                        "in `q_type`)")
    p.add_argument("--tokenizer_path", default="Qwen/Qwen3-VL-30B-A3B-Thinking",
                   help="HF hub id or local path of the tokenizer used to "
                        "compute description token lengths")
    p.add_argument("--len_threshold", type=int, default=20,
                   help="reject pairs whose winner/loser token-length "
                        "difference is >= this many tokens")
    p.add_argument("--seed", type=int, default=42,
                   help="rng seed for shuffling the output records")
    p.add_argument("--report", action="store_true",
                   help="print pos/pos, pos/neg, neg/neg statistics on the "
                        "selected pairs after writing")

    v = subparsers.add_parser("validation", help="sample held-out episodic memories")
    v.add_argument("--video_info_path", required=True,
                   help="jsonl of video metadata records with id/start_time/end_time")
    v.add_argument("--memory_root", required=True,
                   help="root containing {id}/{clip_idx}/{id}_{clip_idx}_{memory_tag}.json")
    v.add_argument("--memory_tag", required=True,
                   help="tag suffix used in per-clip memory filenames")
    v.add_argument("--clip_seconds", type=int, default=10,
                   help="clip length in seconds")
    v.add_argument("--sample_size", type=int, default=576,
                   help="number of records to keep after shuffling")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--output_path", required=True,
                   help="destination .jsonl of sampled records")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "pairs":
        build_pairs(args)
    elif args.task == "validation":
        build_validation(args)

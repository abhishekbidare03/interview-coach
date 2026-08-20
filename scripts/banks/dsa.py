r"""Data Structures & Algorithms.

Replaces the generated DSA bank, which had the same two problems as the
generated Python one: nothing at difficulty 1 at all, and questions phrased as
written-exam prompts rather than as something a person says.

These are spoken questions, so they are about *reasoning* rather than about
writing code — "which structure would you use and why", not "implement a
balanced tree". A candidate cannot type into a microphone, and a question that
needs a whiteboard graded against expected points would grade badly.

Complexity is written the way it is said aloud — "linear time", "log n" — and
never as a symbol, because the reference answer is read out by a speech
synthesiser when you get something wrong.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "arrays and indexing",
     "Why is looking up an array element by index so fast?",
     "Because the elements sit in one contiguous block of memory and all have the "
     "same size, so the address is a single multiplication away from the start. "
     "It takes constant time regardless of how big the array is.",
     ["elements are stored contiguously",
      "every element is the same size",
      "the address is computed arithmetically",
      "lookup is constant time regardless of size"],
     []),

    (1, "arrays versus linked lists",
     "When would a linked list beat an array?",
     "When you insert or remove in the middle a lot. An array has to shift "
     "everything after the insertion point, which is linear. A linked list just "
     "repoints two references. The trade is that you lose fast indexing and "
     "cache locality.",
     ["insertion and deletion in the middle are cheaper",
      "an array must shift subsequent elements",
      "a linked list only updates references",
      "you lose constant-time indexing"],
     ["What does an array give you that a linked list cannot?"]),

    (1, "stacks and queues",
     "What's the difference between a stack and a queue?",
     "The order things come out. A stack is last in, first out — the most "
     "recently added is next. A queue is first in, first out, like a line at a "
     "counter. Both add and remove in constant time.",
     ["a stack is last in, first out",
      "a queue is first in, first out",
      "both add and remove in constant time",
      "gave an example of where each is used"],
     []),

    (1, "hash tables and collision handling",
     "How does a hash table find a value so quickly?",
     "It runs the key through a hash function to get a position in an array and "
     "goes straight there, instead of searching. That is constant time on "
     "average. It costs the ordering — a hash table has no useful order.",
     ["the key is hashed to an array position",
      "lookup goes directly to that position",
      "average time is constant",
      "no ordering is preserved"],
     ["What happens when two keys hash to the same slot?"]),

    (1, "time and space complexity analysis",
     "What does big O notation actually tell you?",
     "How the cost grows as the input grows, ignoring constants and small terms. "
     "It says linear time doubles when the input doubles. It deliberately says "
     "nothing about actual speed — a quadratic algorithm can beat a linear one "
     "on small inputs.",
     ["it describes growth as input size increases",
      "constants and lower-order terms are ignored",
      "it is about scaling, not absolute speed",
      "a worse complexity can win on small inputs"],
     []),

    (1, "binary search and its variants",
     "How does binary search work, and what does it require?",
     "You look at the middle element, and because the data is sorted you know "
     "which half the target must be in, so you discard the other half and repeat. "
     "Halving each step gives log n time. It only works if the data is already "
     "sorted.",
     ["compare against the middle element",
      "discard the half that cannot contain the target",
      "the data must be sorted",
      "it runs in logarithmic time"],
     []),

    (1, "sets and membership",
     "You need to check whether you have seen an item before. What do you use?",
     "A hash set. Membership testing is constant time on average, versus scanning "
     "a list which is linear. Doing that check inside a loop is the difference "
     "between linear and quadratic overall.",
     ["use a hash set",
      "membership testing is constant time on average",
      "scanning a list is linear each time",
      "it avoids a quadratic loop"],
     []),

    (1, "binary trees and traversals",
     "What makes a binary search tree different from an ordinary binary tree?",
     "The ordering rule. Everything in the left subtree is smaller than the node "
     "and everything in the right is larger, all the way down. That invariant is "
     "what lets you discard half the tree at each step when searching.",
     ["left subtree holds smaller values",
      "right subtree holds larger values",
      "the property holds recursively at every node",
      "it enables search by discarding half each step"],
     []),

    (1, "recursion and stack depth",
     "What does every recursive function need to avoid running forever?",
     "A base case that returns without recursing, and each call has to move "
     "towards it. Without both, it recurses until the call stack runs out and you "
     "get a stack overflow.",
     ["a base case that stops the recursion",
      "each call must progress towards the base case",
      "otherwise it recurses until the stack is exhausted",
      "the result is a stack overflow"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "hash tables and collision handling",
     "What is a hash collision, and how is it handled?",
     "Two different keys landing in the same slot, which is unavoidable because "
     "there are more possible keys than slots. Either the slot holds a small list "
     "of entries, or you probe forward for the next free slot. Both keep lookup "
     "constant on average.",
     ["two keys map to the same position",
      "it is unavoidable given finite slots",
      "chaining stores multiple entries per slot",
      "open addressing probes for another slot"],
     ["When does a hash table stop being constant time?"]),

    (2, "sorting algorithm tradeoffs",
     "Why is quicksort usually preferred over merge sort in practice?",
     "It sorts in place, so it does not need a second array, and its memory "
     "access pattern is cache-friendly, which makes it faster in wall-clock terms "
     "despite the same average complexity. Merge sort wins when you need "
     "guaranteed worst case or a stable sort.",
     ["quicksort sorts in place with less extra memory",
      "better cache behaviour makes it faster in practice",
      "both average to n log n",
      "merge sort guarantees worst case and is stable"],
     ["What is quicksort's worst case, and when does it happen?"]),

    (2, "time and space complexity analysis",
     "What's the complexity of looking something up in a sorted array versus a hash table?",
     "The sorted array gives log n with binary search; the hash table gives "
     "constant time on average. The hash table wins on lookup, but the sorted "
     "array supports range queries and ordered traversal, which the hash table "
     "cannot do at all.",
     ["sorted array with binary search is logarithmic",
      "hash table is constant time on average",
      "the hash table is faster for single lookups",
      "the sorted array supports range and ordered queries"],
     []),

    (2, "two pointers and sliding window",
     "What kind of problem does the two pointer technique solve?",
     "One where you would otherwise compare every pair. If the data is sorted or "
     "you are scanning a sequence, two indices moving towards each other or in "
     "the same direction cover the possibilities in one pass, turning quadratic "
     "into linear.",
     ["it avoids checking every pair",
      "two indices move through the data",
      "it usually needs sorted data or a sequence",
      "it reduces quadratic work to linear"],
     []),

    (2, "breadth-first versus depth-first search",
     "What's the practical difference between breadth-first and depth-first search?",
     "Breadth-first explores level by level using a queue, so the first time it "
     "reaches a node is by the shortest path in edges. Depth-first goes as deep "
     "as it can using a stack or recursion, uses less memory on wide graphs, but "
     "gives no shortest-path guarantee.",
     ["breadth-first explores level by level with a queue",
      "it finds the fewest-edges path first",
      "depth-first goes deep using a stack or recursion",
      "depth-first uses less memory on wide graphs"],
     ["Which would you use to find the shortest route, and why?"]),

    (2, "heaps and priority queues",
     "What is a heap useful for?",
     "Repeatedly getting the smallest or largest item without sorting everything. "
     "The extreme is at the root so peeking is constant, and inserting or "
     "removing is log n. That is what a priority queue is built on.",
     ["it gives fast access to the minimum or maximum",
      "the extreme element is at the root",
      "insert and remove are logarithmic",
      "it backs a priority queue"],
     []),

    (2, "arrays versus linked lists",
     "Why can appending to a dynamic array be constant time on average?",
     "Because it does not grow by one each time. When it runs out of space it "
     "allocates a larger block, usually double, and copies. Those copies are rare "
     "and get cheaper per element as the array grows, so the cost averages out to "
     "constant.",
     ["capacity grows by a multiple, not by one",
      "a resize copies all existing elements",
      "resizes become rarer as the array grows",
      "the cost amortises to constant per append"],
     []),

    (2, "graph representations",
     "How would you represent a graph in memory?",
     "An adjacency list for most things — each node keeps its neighbours, so "
     "space is proportional to the edges and iterating a node's neighbours is "
     "fast. An adjacency matrix costs space in the square of the nodes but "
     "answers 'is there an edge' instantly, so it suits dense graphs.",
     ["an adjacency list stores neighbours per node",
      "its space is proportional to the number of edges",
      "a matrix uses space quadratic in the nodes",
      "a matrix gives constant-time edge lookup and suits dense graphs"],
     []),

    (2, "binary trees and traversals",
     "What does an in-order traversal of a binary search tree give you?",
     "The values in sorted order. You visit the left subtree, then the node, then "
     "the right, and because of the ordering invariant that produces ascending "
     "values. It is the cheapest way to get a sorted sequence out of one.",
     ["left subtree, then node, then right subtree",
      "it yields the values in sorted order",
      "it follows from the search tree ordering property",
      "it visits every node once, in linear time"],
     []),

    (2, "recursion and stack depth",
     "When is recursion a bad choice?",
     "When the depth can get large, because each call consumes stack and Python "
     "in particular caps it fairly low. Anything that recurses proportionally to "
     "the input size is safer as a loop with an explicit stack. Deep recursion "
     "also hides its memory cost.",
     ["deep recursion can exhaust the call stack",
      "each call has stack frame overhead",
      "rewrite as a loop with an explicit stack",
      "the memory cost is not obvious in the code"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "balanced trees and when they matter",
     "What goes wrong with an unbalanced binary search tree?",
     "It degenerates into a linked list. Insert already-sorted data and every "
     "node goes down the same side, so searching becomes linear instead of log n "
     "— the exact case you chose a tree to avoid. Balanced variants rotate on "
     "insert to keep the height logarithmic.",
     ["it can degrade into a linked list",
      "sorted insertion order is the common cause",
      "search becomes linear instead of logarithmic",
      "balanced trees restructure to bound the height"],
     []),

    (3, "dynamic programming versus greedy",
     "When does a greedy algorithm fail?",
     "When a locally best choice rules out a better overall solution. Greedy "
     "works only when the problem has the property that local optima compose into "
     "a global one. Otherwise you need dynamic programming, which considers "
     "combinations rather than committing at each step.",
     ["a locally optimal choice can prevent a better global one",
      "greedy requires a specific structural property",
      "dynamic programming considers overlapping alternatives",
      "gave an example where greedy fails"],
     []),

    (3, "dynamic programming versus greedy",
     "What makes a problem suitable for dynamic programming?",
     "Two things: the optimal solution is built from optimal solutions to "
     "subproblems, and those subproblems repeat. The repetition is what makes "
     "caching worth it — without overlap you are just doing recursion with extra "
     "bookkeeping.",
     ["optimal substructure: subproblem solutions compose",
      "overlapping subproblems that recur",
      "results are cached and reused",
      "without overlap, caching gains nothing"],
     ["What's the difference between memoisation and tabulation?"]),

    (3, "shortest path algorithms",
     "Why does breadth-first search stop working for shortest paths once edges have weights?",
     "Because it counts edges, not cost. With weights, a path with more edges can "
     "be cheaper, and breadth-first will settle a node before the cheaper longer "
     "route is found. Dijkstra fixes it by always expanding the lowest total cost "
     "so far, using a priority queue.",
     ["breadth-first minimises edge count, not weight",
      "a longer path can have lower total cost",
      "Dijkstra expands the cheapest known node first",
      "it uses a priority queue rather than a plain queue"],
     ["What breaks Dijkstra?"]),

    (3, "hash tables and collision handling",
     "When does a hash table degrade to linear time?",
     "When too many keys land in the same bucket — a bad hash function, or "
     "adversarial input chosen to collide. Then a lookup walks a long chain. A "
     "high load factor also degrades it, which is why implementations resize once "
     "occupancy passes a threshold.",
     ["when many keys collide into one bucket",
      "a poor hash function or adversarial keys cause it",
      "lookup degenerates to scanning a chain",
      "a high load factor triggers resizing to prevent it"],
     []),

    (3, "sorting algorithm tradeoffs",
     "What does it mean for a sort to be stable, and when does it matter?",
     "That equal elements keep their original relative order. It matters when you "
     "sort by one key after another — sort by name, then stably by department, "
     "and within each department the names are still in order. With an unstable "
     "sort that ordering is lost.",
     ["equal elements retain their original order",
      "it matters when sorting by successive keys",
      "it preserves the earlier ordering within ties",
      "an unstable sort discards that information"],
     []),

    (3, "two pointers and sliding window",
     "What problems does a sliding window solve?",
     "Ones asking about every contiguous subarray or substring of some size or "
     "property. Instead of recomputing each window from scratch, you add what "
     "enters and subtract what leaves, so the whole scan is linear rather than "
     "quadratic.",
     ["it handles contiguous subarrays or substrings",
      "the window updates incrementally as it moves",
      "add the entering element and remove the leaving one",
      "it turns quadratic recomputation into a linear scan"],
     []),

    (3, "time and space complexity analysis",
     "How would you reason about the complexity of a nested loop over the same array?",
     "The outer runs n times and the inner runs up to n for each, so it is "
     "quadratic. If the inner loop starts where the outer is, it is about half "
     "that — still quadratic, since constants drop out. The usual fix is a hash "
     "set or sorting first.",
     ["multiply the iteration counts of the loops",
      "it is quadratic in the input size",
      "halving the inner range does not change the class",
      "a hash set or a sort often reduces it"],
     []),

    (3, "graph representations",
     "How would you detect a cycle in a directed graph?",
     "Depth-first search, tracking which nodes are on the current path rather "
     "than merely visited. Reaching a node already on the path means a cycle. "
     "Just checking 'visited' is not enough — that only tells you it was seen on "
     "some earlier branch.",
     ["use depth-first search",
      "track nodes on the current recursion path",
      "an edge back to a node on the path is a cycle",
      "a plain visited set is insufficient"],
     []),

    (3, "arrays versus linked lists",
     "Why are arrays often faster than linked lists even when complexity says otherwise?",
     "Cache locality. Array elements sit together, so reading one pulls its "
     "neighbours into cache and the next access is nearly free. Linked list nodes "
     "are scattered, so each step is a potential cache miss — and a miss costs far "
     "more than the pointer chase itself.",
     ["arrays are contiguous in memory",
      "reading one element caches its neighbours",
      "linked list nodes are scattered and cause cache misses",
      "complexity ignores this constant factor"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "shortest path algorithms",
     "Why does Dijkstra break with negative edge weights?",
     "Because it commits. Once it settles a node as final it never revisits it, "
     "and that is only safe if adding an edge can never reduce a total. A "
     "negative edge can make an already-settled path cheaper later, so the "
     "answer is wrong. Bellman-Ford relaxes repeatedly instead.",
     ["it finalises a node once and never revisits it",
      "that assumes costs only increase along a path",
      "a negative edge can improve a settled distance",
      "Bellman-Ford handles negatives by repeated relaxation"],
     []),

    (4, "time and space complexity analysis",
     "What's the difference between average and amortised complexity?",
     "Average is over a distribution of inputs — it is a probabilistic claim, and "
     "a bad input can be slow. Amortised is a guarantee over a sequence of "
     "operations: any run of n operations costs the bound, even though individual "
     "ones vary.",
     ["average is over a distribution of inputs",
      "amortised is over a sequence of operations",
      "amortised gives a guarantee, average does not",
      "dynamic array append is amortised constant"],
     []),

    (4, "hash tables and collision handling",
     "How would you find the first non-repeating character in a stream?",
     "Two passes over stored state rather than over the stream twice. Keep counts "
     "in a hash map, and keep insertion order — either an ordered map or a queue "
     "of candidates whose front is discarded once its count exceeds one. That "
     "gives an answer at any point in linear time.",
     ["count occurrences in a hash map",
      "preserve insertion order of candidates",
      "discard candidates whose count exceeds one",
      "it answers in linear time over the stream"],
     []),

    (4, "dynamic programming versus greedy",
     "How do you reduce the memory of a dynamic programming solution?",
     "By noticing how far back the recurrence actually reaches. If each row "
     "depends only on the previous one, you keep two rows instead of the whole "
     "table — linear space instead of quadratic. You lose the ability to "
     "reconstruct the path, which sometimes matters.",
     ["check how many previous states the recurrence needs",
      "keep only those rows rather than the full table",
      "it reduces quadratic space to linear",
      "you lose the ability to reconstruct the solution path"],
     []),

    (4, "sorting algorithm tradeoffs",
     "You need the ten largest items from a billion. How?",
     "Not by sorting. Keep a heap of size ten and stream the data past it, "
     "replacing the smallest whenever something larger arrives. That is linear "
     "time in the data and constant memory, versus sorting the whole billion.",
     ["do not sort the entire dataset",
      "maintain a heap of the k best seen",
      "replace the smallest when a larger value arrives",
      "linear time and memory proportional to k"],
     []),

    (4, "balanced trees and when they matter",
     "When would you choose a balanced tree over a hash table?",
     "When you need order. A tree gives sorted traversal, range queries, and "
     "nearest-key lookups, none of which a hash table supports. It also has a "
     "worst-case guarantee rather than an average, which matters if the input "
     "might be adversarial.",
     ["when ordering or range queries are needed",
      "a hash table has no useful ordering",
      "trees support nearest-key and successor queries",
      "trees give worst-case rather than average guarantees"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "time and space complexity analysis",
     "How would you approach a problem where the obvious solution is too slow?",
     "First work out what the obvious one costs and what the target is, because "
     "that gap tells you what to look for. Then the usual levers: precompute "
     "with a hash map, sort to enable two pointers or binary search, cache "
     "overlapping subproblems, or find structure in the input that lets you skip "
     "work.",
     ["quantify the current cost and the required cost",
      "the gap indicates what complexity class is needed",
      "trade memory for time with a hash map or cache",
      "sorting can unlock a faster technique"],
     []),

    (5, "graph representations",
     "How would you handle a graph too large to fit in memory?",
     "Stop holding it all. Stream the edges and process in passes, partition by "
     "node so each part fits and handle cross-partition edges explicitly, or keep "
     "it on disk with an index and accept the access cost. Which one depends on "
     "whether the algorithm needs random access or only local neighbourhoods.",
     ["stream or process the graph in passes",
      "partition it so each part fits in memory",
      "handle edges crossing partitions explicitly",
      "the choice depends on the access pattern needed"],
     []),

    (5, "sorting algorithm tradeoffs",
     "How would you sort more data than fits in memory?",
     "External merge sort. Read as much as fits, sort it, write it out as a run, "
     "and repeat — then merge the runs together, reading a block from each. The "
     "cost is dominated by disk passes, so you merge as many runs at once as "
     "buffer space allows.",
     ["sort chunks that fit in memory and write them out",
      "merge the sorted runs together afterwards",
      "only a block from each run is held at once",
      "minimise passes over the data, since disk dominates"],
     []),
]

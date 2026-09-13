---
license: mit
---

# GraphWalks: a multi hop reasoning long context benchmark

In Graphwalks, the model is given a graph represented by its edge list and asked to perform an operation. 

Example prompt:

```
You will be given a graph as a list of directed edges. All nodes are at least degree 1. 
You will also get a description of an operation to perform on the graph.
Your job is to execute the operation on the graph and return the set of nodes that the operation results in. 
If asked for a breadth-first search (BFS), only return the nodes that are reachable at that depth, do not return the starting node.
 If asked for the parents of a node, only return the nodes that have an edge leading to the given node, do not return the given node itself.
The graph has the following edges:
uvwx -> alke
abcd -> uvwx
abcd -> efgh
efgh -> uvwx

Example 1:
Operation:
Perform a BFS from node abcd with depth 1.
Final Answer: [uvwx, efgh]

```

## Data schema

|column|description|
|------|-----------|
|`prompt`| A 3-shot example followed by the graph and the operation to be performed. This is mean to be supplied to the model as a user message.|
|`answer`| A list of node ids that the model should respond with.|
|`prompt_chars`| The number of characters in the prompt.|
|`problem_type`| Either `bfs` or `parents` for the graph operation requested|

## Extraction and Grading

We use the following code to extract answers from responses

```python
def get_list(self, response: str) -> tuple[list[str], bool]:
        # get the very last line of the response
        line = response.split("\n")[-1]
        # check if formatted correctly
        if "Final Answer:" not in line:
            return [], True
        list_part = re.search(r"Final Answer: ?\[.*\]", line)
        if list_part:
            result_list = list_part.group(0).strip("[]").split(",")
            # if the list was empty, then get [] not [""]
            result_list = [item.strip() for item in result_list if item.strip()]
            return result_list, False
        else:
            return [], True
```

We grade each example with the following

```python
n_overlap = len(sampled_set & truth_set)
recall = n_overlap / n_golden if n_golden > 0 else 0
precision = n_overlap / n_sampled if n_sampled > 0 else 0
f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 1
```

## OpenAI results

Please refer to the [GPT 4.1 blog post](https://openai.com/index/gpt-4-1/).

## Changelong
- 4/12/2025: Initial dataset published
- 2/27/26: Bugfix: A bug during generation led to 24/400 parents samples in `graphwalks_128k_and_shorter.parquet` to contain the incorrect ground truth - the root node was inadvertently included.
Additionally, in BFS there was ambiguity in the prompt - during normal BFS, revisited nodes are not added to the frontier.
However, as worded, the model is asked to "return the nodes that are reachable at that depth", which would imply including revisited nodes. The prompt has been modified to specify that only nodes at exactly the desired depth should be returned.
Thank you to the Claude Opus 4.6 system card for pointing out the issue in the parent samples!

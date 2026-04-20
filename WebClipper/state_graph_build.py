import json
import time
import copy
import os
import argparse
from collections import defaultdict
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EXTRACTOR_API_KEY = os.getenv("EXTRACTOR_API_KEY")
EXTRACTOR_BASE_URL = os.getenv("EXTRACTOR_BASE_URL")

if not EXTRACTOR_API_KEY or not EXTRACTOR_BASE_URL:
    raise RuntimeError(
        "environment variable EXTRACTOR_API_KEY or EXTRACTOR_BASE_URL is not detected.\n"
        "Please create a .env file in the project root directory and set:\n"
        "EXTRACTOR_API_KEY=Your API_KEY\n"
        "EXTRACTOR_BASE_URL=Your API_BASE_URL\n"
    )

MODEL_NAME = "auto"
MAX_WORKERS = 20
MAX_VALIDATION_RETRIES = 10
NUM_REPETITIONS = 3
MAX_RETRIES = 5

def number_assistant_roles_new_list(messages):
    new_messages = []
    assistant_counter = 1
    for message in messages:
        new_message = message.copy()
        if new_message['role'] == 'assistant':
            new_message['role'] = f'assistant{assistant_counter}'
            assistant_counter += 1
        new_messages.append(new_message)
    return new_messages

def chat_with_model(messages, model=MODEL_NAME, max_retries=50):
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=EXTRACTOR_API_KEY, base_url=EXTRACTOR_BASE_URL)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=600,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt % 2 == 0:
                 print(f"API Error: {e}. Retrying API call ({attempt + 1}/{max_retries})...")
            time.sleep(1)
    print(f"\n[!] Critical: Failed to get response from model after {max_retries} attempts.")
    return None

def replace_mask_with_reason(content):
    if isinstance(content, str):
        return content.replace("<think>", "<reason>").replace("</think>", "</reason>")
    return content

ITERATIVE_ACTION_NODE_INSTRUCTION = """## Objective: Extract an Action Node from a Single Assistant Turn

**Task Overview:** You will receive a separate conversation history and a "current Assistant turn" to be analyzed. Your task is to **focus solely on analyzing the "current Assistant turn"**, summarize it into a single Action Node, and output it in the format of a **single JSON object**.

---

### 1. Conversation Content

#### 1.1 Conversation History (Context)
<history>
[YOUR_HISTORY]
</history>

#### 1.2 Current Assistant Turn to be Analyzed
<current_turn>
[YOUR_CURRENT_TURN]
</current_turn>

---

### 2. JSON Output Format Requirements

The final output must strictly follow the structure of the **single JSON object** below. **Do not** include any lists or outer wrappers for the `Action Node`.

{"Action":"<action_type>","Goal":"<goal_summary>"}

For example:
{"Action":"Search","Goal":"Get the statistical data for China's urban population in 1949 and 2009"}
{"Action":"PythonInterpreter","Goal":"Calculate the Compound Annual Growth Rate (CAGR) based on the population data"}
{"Action":"Visit","Goal":"Verify the official source for the 1949 urban population data"}
{"Action":"Answer","Goal":"Provide the answer: approximately 4.04%"}

---

### 3. Action Node (A-Node) Extraction Rules

1.  **Action (Core Rule):**
    *   If the `content` field of the **"Current Assistant Turn to be Analyzed" contains the special tag `<answer>`**, the `Action` must be assigned the value `"Answer"`.
    *   Otherwise, assign the `Action` as one of `Search`, `Visit`, or `PythonInterpreter` based on the tool it calls.

2.  **Goal:**
    *   For non-`Answer` actions: Summarize the purpose of the tool call in a **concise description**. This summary should cover the search queries for a Search, the goal for a Visit, or the calculation's purpose for a PythonInterpreter.
    *   For an `Answer` action: The `Goal` field must be "Provide the answer: " + **the extracted core short answer** (a word or a short phrase), which is the minimal piece of information that directly answers the original user's question.

---

Now, based on the requirements above, process the provided "Current Assistant Turn to be Analyzed":

Output:
"""

def extract_single_action_node(conversation_history, current_assistant_turn):
    history_for_prompt = copy.deepcopy(conversation_history)
    turn_for_prompt = copy.deepcopy(current_assistant_turn)

    for msg in history_for_prompt:
        if msg.get('role', '').startswith('assistant'):
            msg['content'] = replace_mask_with_reason(msg.get('content'))

    turn_for_prompt['content'] = replace_mask_with_reason(turn_for_prompt.get('content'))

    prompt = ITERATIVE_ACTION_NODE_INSTRUCTION.replace(
        '[YOUR_HISTORY]', json.dumps(history_for_prompt, ensure_ascii=False)
    ).replace(
        '[YOUR_CURRENT_TURN]', json.dumps(turn_for_prompt, ensure_ascii=False)
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(MAX_VALIDATION_RETRIES):
        raw_result = chat_with_model(messages, max_retries=50)
        if raw_result is None:
            return None

        cleaned_result_str = raw_result.split('</think>')[-1].strip().replace('```json', '').replace('```', '').strip()

        try:
            parsed_node = json.loads(cleaned_result_str)

            if not all(key in parsed_node for key in ["Action", "Goal"]):
                if attempt < MAX_VALIDATION_RETRIES - 1:
                    print(f"Node Validation Failed (Attempt {attempt + 1}/{MAX_VALIDATION_RETRIES}): Missing required keys. Retrying...")
                continue

            return parsed_node

        except json.JSONDecodeError:
            if attempt < MAX_VALIDATION_RETRIES - 1:
                print(f"Node Validation Failed (Attempt {attempt + 1}/{MAX_VALIDATION_RETRIES}): Invalid JSON format. Retrying...")
            continue

    print(f"\n[!] Critical: Failed to extract a valid single node after {MAX_VALIDATION_RETRIES} attempts.")
    return None

def process_action_extraction_tasks(all_tasks):
    reassembled_data = defaultdict(lambda: {"nodes": [], "failed": False})

    def process_single_node_task(task):
        node = extract_single_action_node(task['history'], task['current_turn'])
        return {
            "item_id": task['item_id'],
            "turn_index": task['turn_index'],
            "node": node
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = executor.map(process_single_node_task, all_tasks)
        for result in tqdm(future_map, total=len(all_tasks), desc="Extracting Action Nodes"):
            item_id = result['item_id']

            if result['node'] is None:
                reassembled_data[item_id]['failed'] = True
                continue

            if reassembled_data[item_id]['failed']:
                continue

            reassembled_data[item_id]['nodes'].append((result['turn_index'], result['node']))

    return reassembled_data

def extract_action_nodes(input_path):
    print(f"\n--- Starting Action Node Extraction for: {input_path} ---")

    item_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item_list.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping a malformed JSON line in {input_path}")
                continue

    if not item_list:
        print(f"Warning: No valid items to process in {input_path}. Skipping.")
        return []

    new_item_list = []
    for item in item_list:
        original_message_list = item.get('messages', [])
        if not original_message_list:
            print(f"Warning: Skipping item with no 'messages' key in {input_path}")
            continue
        numbered_list = number_assistant_roles_new_list(original_message_list)
        new_item_list.append(numbered_list)

    all_tasks = []
    for item_id, item_content in enumerate(new_item_list):
        history_start_index = 1 if item_content and item_content[0].get('role') == 'system' else 0

        assistant_turns = [(i, msg) for i, msg in enumerate(item_content) if msg['role'].startswith('assistant')]
        if not assistant_turns:
            continue

        for turn_idx, (original_idx, assistant_msg) in enumerate(assistant_turns):
            history = item_content[history_start_index:original_idx]
            task = {
                "item_id": item_id,
                "turn_index": turn_idx + 1,
                "history": history,
                "current_turn": assistant_msg,
            }
            all_tasks.append(task)

    print(f"Flattened all conversations into {len(all_tasks)} individual extraction tasks.")

    reassembled_data = process_action_extraction_tasks(all_tasks)

    print("\nReassembling results and performing final validation...")
    action_results = []
    num_failed_items = 0
    total_original_items = len(new_item_list)

    for item_id in range(total_original_items):
        if item_id not in reassembled_data:
            num_failed_items += 1
            continue

        data = reassembled_data[item_id]
        if data['failed']:
            num_failed_items += 1
            continue

        sorted_nodes = sorted(data['nodes'], key=lambda x: x[0])

        answer_node_count = sum(1 for _, node in sorted_nodes if node.get("Action") == "Answer")
        if answer_node_count > 1:
            print(f"\n[!] Final Validation Failed for item {item_id}: Found {answer_node_count} 'Answer' nodes. Skipping item.")
            num_failed_items += 1
            continue

        action_node_list = [{f"A{turn_idx}": node} for turn_idx, node in sorted_nodes]
        final_result_obj = {"Action Node": action_node_list}

        result_dict = {
            "raw_message": new_item_list[item_id],
            "state_graph_result": json.dumps(final_result_obj, ensure_ascii=False)
        }
        action_results.append(result_dict)

    if num_failed_items > 0:
        print(f"\nSkipped {num_failed_items} complete conversations out of {total_original_items} due to API errors or validation failures.")

    print(f"Successfully extracted action nodes for {len(action_results)} conversations.")
    return action_results

FEW_SHOT_EXAMPLE_1_CONTEXT = {
    "graph_workspace": {
        "Information Node": [{"I0": {"category": "Original Query", "info": "From 1949 to 2009, what was the average annual growth rate of China's urban population?"}}],
        "Action Node": [{"A1": {"Action": "Search","Goal": "Get statistical data for China's urban population in 1949 and 2009"}}],
        "Edge": [["I0", "A1"]]
    },
    "previous_action": {"A1": {"Action": "Search","Goal": "Get statistical data for China's urban population in 1949 and 2009"}},
    "observation": "Search results show...[National Bureau of Statistics]...Urban population (10,000 people), nationwide, 1949, 5765 ...and...[Sina Finance]...In 2009, China's urban population reached 622 million people according to statistical caliber...",
    "current_action": {"A2": {"Action": "PythonInterpreter","Goal": "Calculate the Compound Annual Growth Rate (CAGR) based on the population data"}},
    "next_info_node_id": "I1"
}

FEW_SHOT_EXAMPLE_1_OUTPUT_JSON = json.dumps({
    "new_information_nodes": [
        {"I1": {"category": "Data", "info": "China's urban population in 1949 was 57.65 million"}},
        {"I2": {"category": "Data", "info": "China's urban population in 2009 was 622 million"}}
    ],
    "new_edges": [["A1", "I1"], ["A1", "I2"], ["I1", "A2"], ["I2", "A2"]]
}, indent=4, ensure_ascii=False)

FEW_SHOT_EXAMPLE_2_CONTEXT = {
    "graph_workspace": {
        "Information Node": [
            {"I0": {"category": "Original Query", "info": "From 1949 to 2009, what was the average annual growth rate of China's urban population?"}},
            {"I1": {"category": "Data", "info": "China's urban population in 1949 was 57.65 million"}},
            {"I2": {"category": "Data", "info": "China's urban population in 2009 was 622 million"}}
        ],
        "Action Node": [
            {"A1": {"Action": "Search","Goal": "Get statistical data for China's urban population in 1949 and 2009"}},
            {"A2": {"Action": "PythonInterpreter","Goal": "Calculate the Compound Annual Growth Rate (CAGR) based on the population data"}},
            {"A3": {"Action": "Visit","Goal": "Verify the official source for the 1949 urban population data"}}
        ],
        "Edge": [["I0", "A1"], ["A1", "I1"], ["A1", "I2"], ["I1", "A2"], ["I2", "A2"], ["I1", "A3"]]
    },
    "previous_action": {"A3": {"Action": "Visit","Goal": "Verify the official source for the 1949 urban population data"}},
    "observation": "According to the National Bureau of Statistics data, China's 'urban population (10,000 people)' in 1949 was 5765, which is 57.65 million.",
    "current_action": {"A4": {"Action": "Visit","Goal": "Verify the official source for the 2009 urban population data"}},
    "next_info_node_id": "I3"
}

FEW_SHOT_EXAMPLE_2_OUTPUT_JSON = json.dumps({
    "new_information_nodes": [],
    "new_edges": [["A3", "I1"], ["I2", "A4"]]
}, indent=4, ensure_ascii=False)

ITERATIVE_PROMPT_TEMPLATE = """## Task: Incremental Graph Construction

You are a graph construction assistant. Your task is to update a state graph based on the provided context.
The context I provide contains the following parts:

#### Context
<previous_action>
The action from the previous round
</previous_action>

<observation>
Information obtained from the external environment based on the previous action
</observation>

<current_action>
The action of the current round
</current_action>

#### Graph Workspace
<graph_workspace>
The current state graph
</graph_workspace>
---

### 1. Your Task

Strictly follow the requirements below to generate a JSON object containing `new_information_nodes` and `new_edges`.

#### Step A. Determine whether to construct new information nodes (new_information_nodes)

1.  **Read and Compare:** Carefully read the content of `<observation>` and compare it semantically with **all** existing `Information Node`s in `<graph_workspace>`.

2.  **Empty Information Handling:** If `<observation>` does not provide any new information, return an empty list `[]` for `new_information_nodes`.

3.  **New Information Extraction:** If new information exists, **only extract** information that is **semantically new, independent, and concise**. During extraction, you must strictly follow these **Information Node Extraction Rules**:
    *   **3.1. Independence and Conciseness:**
        *   The `info` field must be a **concise short phrase** that is **semantically independent and unambiguous**.
        *   **Do not** use pronouns or referential relationships.
    *   **3.2. Uniqueness and Comprehensiveness:**
        *   Across the entire `Information Node` list, the values of all `info` fields must be **semantically unique**.
        *   **Mandatory Requirement:** Perform **merging and refining** of **similar or redundant information** to ensure no duplicate or redundant entries in the final list.
    *   **3.3. Relevance to Original Query:**
        *   The extraction direction of `Information Node` should consider relevance to the original question (i.e., the content of node `I0`). For example, when the original question emphasizes information sources, you need to include corresponding source information in the `info` field of the Information Node.
    *   **3.4. Categories:**
        *   Other I-Node categories are limited to: `Data`, `Opinion`, `Event`. Use `General` for difficult cases.

4.  **Assign New IDs:** If new information nodes need to be assigned, **assign IDs for new nodes**:
    *   You must start numbering new nodes from the given starting ID **`{next_info_node_id}`**.
    *   If multiple new nodes are extracted, **strictly increment IDs sequentially** (e.g., if starting from `I3`, new nodes must be `I3`, `I4`, ...).

#### Step B. Construct new edges (new_edges)

You need to create two types of edges:

1.   **`Action -> Info` Connection:** Create edges from `<previous_action>` (ID: `{previous_action_id}`) to **all** information nodes it directly produces. If Step A generated new information, edges from `<previous_action>` to the new information nodes need to be created here; if Step A did not generate new information, or if `<observation>` contains information already existing in `<workspace>`, then create edges from `<previous_action>` to the already existing `Information Node` in `<workspace>`.
2.   **`Info -> Action` Connection:** From **all** information nodes in `<graph_workspace>` (including the new nodes you just extracted), find **all** nodes that serve as the decision basis, directly leading to the action `<current_action>` (ID: `{current_action_id}`), and create edges to it.

* **Final Answer Convergence Constraint:**
      * **Edges pointing to Answer nodes (i.e., where the assistant's content contains the <answer> token) are only allowed to connect to edges that directly contribute to the answer in the Goal.

* **Note:** The graph will form a complex network structure, and **cycles** may occur. Faithfully record all relevant edges.

#### C. Output Format
Strictly output in the following JSON format. Do not include any other explanations.

---

### 2. Task Execution Examples

#### Example 1: Finding New Information

#### Context
<previous_action>
{fs1_previous_action_json}
</previous_action>

<observation>
{fs1_observation_content}
</observation>

<current_action>
{fs1_current_action_json}
</current_action>

#### Graph Workspace
<graph_workspace>
{fs1_workspace_json}
</graph_workspace>

Output:
```json
{few_shot_example_1_output}
```

---

#### Example 2: No New Information Found (Information Redundancy)

#### Context
<previous_action>
{fs2_previous_action_json}
</previous_action>

<observation>
{fs2_observation_content}
</observation>

<current_action>
{fs2_current_action_json}
</current_action>

#### Graph Workspace
<graph_workspace>
{fs2_workspace_json}
</graph_workspace>

Output:
```json
{few_shot_example_2_output}
```

---

### 3. Actual Current Task

Now, generate output for the following actual task.

#### Context
<previous_action>
{previous_action_json}
</previous_action>

<observation>
{current_observation_content}
</observation>

<current_action>
{current_action_json}
</current_action>

#### Graph Workspace
<graph_workspace>
{workspace_json}
</graph_workspace>


Output:
"""

def create_trajectory_steps(original_messages, action_node_list):
    steps = []
    assistant_msgs = [msg for msg in original_messages if msg.get('role', '').startswith('assistant')]
    observations = []

    for i in range(len(assistant_msgs)):
        try:
            current_msg_index = -1
            for idx, msg in enumerate(original_messages):
                if msg == assistant_msgs[i]:
                    current_msg_index = idx
                    break
            if current_msg_index == -1:
                raise ValueError("Assistant message not found")

            observation = next(msg for msg in original_messages[current_msg_index+1:] if msg.get('role') != 'system')
            observations.append(observation)
        except (ValueError, StopIteration):
            observations.append(None)

    if len(action_node_list) != len(observations):
        print(f"Warning: Mismatch actions ({len(action_node_list)}) vs observations ({len(observations)}).")

    for i in range(len(action_node_list) - 1):
        previous_action = action_node_list[i]
        observation = observations[i]
        current_action = action_node_list[i+1]
        observation_content = "No observation was returned." if observation is None else observation['content']
        steps.append((previous_action, observation_content, current_action))

    return steps

def create_iterative_prompt(workspace, prev_action, obs_content, curr_action, next_info_node_id):
    fs1 = FEW_SHOT_EXAMPLE_1_CONTEXT
    fs2 = FEW_SHOT_EXAMPLE_2_CONTEXT

    return ITERATIVE_PROMPT_TEMPLATE.format(
        workspace_json=json.dumps(workspace, ensure_ascii=False, indent=2),
        previous_action_json=json.dumps(prev_action, ensure_ascii=False, indent=2),
        current_observation_content=obs_content,
        current_action_json=json.dumps(curr_action, ensure_ascii=False, indent=2),
        previous_action_id=list(prev_action.keys())[0],
        current_action_id=list(curr_action.keys())[0],
        next_info_node_id=next_info_node_id,

        fs1_workspace_json=json.dumps(fs1['graph_workspace'], ensure_ascii=False, indent=2),
        fs1_previous_action_json=json.dumps(fs1['previous_action'], ensure_ascii=False, indent=2),
        fs1_observation_content=fs1['observation'],
        fs1_current_action_json=json.dumps(fs1['current_action'], ensure_ascii=False, indent=2),
        few_shot_example_1_output=FEW_SHOT_EXAMPLE_1_OUTPUT_JSON,

        fs2_workspace_json=json.dumps(fs2['graph_workspace'], ensure_ascii=False, indent=2),
        fs2_previous_action_json=json.dumps(fs2['previous_action'], ensure_ascii=False, indent=2),
        fs2_observation_content=fs2['observation'],
        fs2_current_action_json=json.dumps(fs2['current_action'], ensure_ascii=False, indent=2),
        few_shot_example_2_output=FEW_SHOT_EXAMPLE_2_OUTPUT_JSON,
    )

def process_item_iteratively(item):
    try:
        original_messages = item.get('raw_message')
        if not original_messages:
            return None

        action_node_list_str = item.get('state_graph_result')
        if not action_node_list_str:
            return None

        action_node_list = json.loads(action_node_list_str).get("Action Node")
        if not action_node_list or len(action_node_list) < 1:
            return None

        paired_steps = create_trajectory_steps(original_messages, action_node_list)

        query_msg = next(msg for msg in original_messages if msg.get('role') == 'user')
        query_info = query_msg['content']
        first_action = action_node_list[0]
        first_action_id = list(first_action.keys())[0]

        graph_workspace = {
            "Information Node": [{"I0": {"category": "Original Query", "info": query_info}}],
            "Action Node": [first_action],
            "Edge": [["I0", first_action_id]]
        }
        info_node_counter = 1

        for step_idx, step in enumerate(paired_steps):
            previous_action, observation_content, current_action = step

            next_info_node_id = f"I{info_node_counter}"
            prompt = create_iterative_prompt(
                graph_workspace,
                previous_action,
                observation_content,
                current_action,
                next_info_node_id
            )

            llm_output = None
            for attempt in range(1, MAX_RETRIES + 1):
                raw_response = chat_with_model([{"role": "user", "content": prompt}], max_retries=500)
                if not raw_response:
                    if attempt == MAX_RETRIES:
                        print(f"\n[!] API call failed permanently in step {step_idx}.")
                    time.sleep(1)
                    continue

                cleaned_response = raw_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                try:
                    parsed_response = json.loads(cleaned_response)
                    if "new_information_nodes" in parsed_response and "new_edges" in parsed_response:

                        new_nodes = parsed_response["new_information_nodes"]
                        is_id_sequence_valid = True
                        expected_start_id_num = int(next_info_node_id[1:])

                        for i, node_dict in enumerate(new_nodes):
                            if not node_dict or not isinstance(node_dict, dict):
                                print(f"\n[!] Validation Failed (Invalid node format). Attempt {attempt}/{MAX_RETRIES}.")
                                is_id_sequence_valid = False
                                break

                            actual_id = list(node_dict.keys())[0]
                            expected_id = f"I{expected_start_id_num + i}"

                            if actual_id != expected_id:
                                print(f"\n[!] Validation Failed (Incorrect ID Sequence). Expected: {expected_id}, Got: {actual_id}. Attempt {attempt}/{MAX_RETRIES}.")
                                is_id_sequence_valid = False
                                break

                        if is_id_sequence_valid:
                            llm_output = parsed_response
                            break

                    else:
                        print(f"\n[!] Validation Failed (Missing keys) in step {step_idx}. Attempt {attempt}/{MAX_RETRIES}.")
                except json.JSONDecodeError:
                    print(f"\n[!] Validation Failed (Invalid JSON) in step {step_idx}. Attempt {attempt}/{MAX_RETRIES}.")

                time.sleep(1)

            if llm_output is None:
                print(f"\n[!!] CRITICAL: Item failed after {MAX_RETRIES} retries in step {step_idx} due to validation errors. Skipping item.")
                return None

            new_nodes_from_llm = llm_output.get("new_information_nodes", [])
            graph_workspace["Information Node"].extend(new_nodes_from_llm)

            new_edges = llm_output.get("new_edges", [])
            graph_workspace["Edge"].extend(new_edges)

            graph_workspace["Action Node"].append(current_action)

            info_node_counter += len(new_nodes_from_llm)

        final_graph_string = json.dumps(graph_workspace, ensure_ascii=False)
        return {"raw_message": item.get('raw_message'), "state_graph_result": final_graph_string}

    except Exception as e:
        print(f"\n[!!!] UNHANDLED EXCEPTION in process_item_iteratively: {e}.")
        import traceback
        traceback.print_exc()
        return None

def construct_info_nodes(action_results):
    print(f"\n--- Starting Info Node Construction ---")
    print(f"Processing {len(action_results)} items with {NUM_REPETITIONS} repetitions each.")

    repeated_item_list = [item for item in action_results for _ in range(NUM_REPETITIONS)]
    total_tasks = len(repeated_item_list)

    print(f"Total tasks to execute: {total_tasks}")
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = executor.map(process_item_iteratively, repeated_item_list)
        all_results = list(tqdm(future_map, total=total_tasks, desc="Constructing Info Nodes"))

    successful_results = [res for res in all_results if res is not None]
    num_failed = total_tasks - len(successful_results)

    if num_failed > 0:
        print(f"\nSkipped {num_failed} tasks due to errors.")

    print(f"Successfully constructed info nodes for {len(successful_results)} tasks.")
    return successful_results

def build_graph_end_to_end(input_path, output_path):
    print("=" * 80)
    print("Starting End-to-End Graph Construction")
    print("=" * 80)

    action_results = extract_action_nodes(input_path)

    if not action_results:
        print("No action nodes extracted. Exiting.")
        return

    final_results = construct_info_nodes(action_results)

    if not final_results:
        print("No info nodes constructed. Exiting.")
        return

    print(f"\nWriting {len(final_results)} final results to {output_path}...")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"--- End-to-End Graph Construction Complete ---")
    print(f"Results written to: {output_path}")
    print("=" * 80)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct the dialogue state graph: Extract Action Nodes from the input JSONL and iterate to generate Info Nodes."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input JSONL File Path"
    )
    parser.add_argument(
        "--output", "-o",
        nargs="+",
        required=True,
        help="Output JSONL File Path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = args.input
    output_paths = args.output

    if len(input_paths) != len(output_paths):
        raise ValueError(
            f"The number of input files ({len(input_paths)}) is different with the number of output files ({len(output_paths)})"
        )

    print(f"Found {len(input_paths)} file pair(s) to process.")

    for input_path, output_path in zip(input_paths, output_paths):
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}. Skipping this pair.")
            continue

        print(f"\nProcessing: {input_path}")
        print(f"Output will be saved to: {output_path}")

        build_graph_end_to_end(input_path, output_path)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
import json
from collections import deque, Counter
import re
import os
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

REWRITER_API_KEY = os.getenv("REWRITER_API_KEY")
REWRITER_BASE_URL = os.getenv("REWRITER_BASE_URL")
PPL_MODEL_PATH = os.getenv("PPL_MODEL_PATH")

if not REWRITER_API_KEY or not REWRITER_BASE_URL:
    raise RuntimeError(
        "environment variable REWRITER_API_KEY or REWRITER_BASE_URL is not detected.\n"
        "Please create a .env file in the project root directory and set:\n"
        "REWRITER_API_KEY=Your API_KEY\n"
        "REWRITER_BASE_URL=Your API_BASE_URL\n"
    )

if not PPL_MODEL_PATH:
    raise RuntimeError(
        "PPL_MODEL_PATH is not set\n"
    )

MODEL_NAME = "auto"
MAX_WORKERS = 10

NUM_CANDIDATES = 3
MAX_FORMAT_RETRIES = 3
MAX_API_RETRIES = 5

ppl_model = None
ppl_tokenizer = None
ppl_device = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTION = """You are an expert AI assistant content editor. Your task is to refine and condense the *thinking process* provided to you, based on the dialogue history.
I will give you three parts:
1. **Dialogue History**: The necessary messages that have been stored so far
2. **Skipped Messages**: Messages between the previous necessary message and current necessary message that are judged to provide no information gain
3. **Current Action to Refine**: The current assistant's thought process and actions that need to be refined

Your task is to modify the current assistant's thought process so that it connects smoothly with the previous necessary message (in Dialogue History), while skipping the intermediate messages that provide no information gain.

Follow these rules strictly:

1. **Primary Goal**: Refine the "Current Action to Refine" to make it more smoothly connect with the last message in "Dialogue History".
2. **Remove Redundancy**: Delete any repetitive thoughts, circular reasoning, or unnecessary self-corrections.
3. **CRITICAL RULE: MERGE SKIPPED STEPS**: The "Skipped Messages" have been identified as providing no information gain. Your refined thought process should seamlessly bridge from the last necessary message to the current action, as if those intermediate steps never happened. 
    - Please do not mention the assistant's action or the user's returned content in "Skipped Messages". 
    - There is one exception: if the search in "Skipped Messages" returns 0 results, but the current action retryes without changing direction, then when refining the current action, you can refer to the assistant's thought process in "Skipped Messages," as such an empty search might be due to external environmental fluctuations.
4. **Maintain Style**: The refined content should maintain the original thinking style - use first-person perspective, self-questioning and logical reasoning similar to the original.
5. **No Hallucination**: Do not introduce any content that doesn't exist in the "Dialogue History" or logical inferences from the skipped steps.
6. **CRITICAL RULE: PRESERVE TOOL CALLS**:
    - If the "Current Action to Refine" contains a <tool_call> section, you must only refine the <think> section. Your output should only be the refined <think> section, without the <tool_call>.
    - The <tool_call> section will be appended to your output automatically, so do not include it.
7. **SPECIAL RULE FOR FINAL ANSWERS**:
    - If the "Current Action to Refine" contains an <answer> section, you should refine both the <think> and <answer> sections.
    - **Crucially, ensure the explanation only uses information and data points that are explicitly present in the "Previous Context".** Remove any claims or data in the explanation that were not previously derived or mentioned.
    - **DO NOT change the core result inside the <answer> tags.**

Your output should ONLY be the refined content, with no extra explanations from you.
"""

def find_minimal_action_subgraph(graph_data: dict) -> list[str]:
    try:
        action_nodes_raw = graph_data.get("Action Node", [])
        info_nodes_raw = graph_data.get("Information Node", [])
        edges = graph_data.get("Edge", [])

        action_node_map = {list(node.keys())[0]: list(node.values())[0] for node in action_nodes_raw}
        info_node_map = {list(node.keys())[0]: list(node.values())[0] for node in info_nodes_raw}

        start_node = None
        for node_id, details in info_node_map.items():
            if details.get("category") == "原始Query":
                start_node = node_id
                break
        
        if not start_node:
            return []

        target_nodes = [node_id for node_id, details in action_node_map.items() if details.get("Action") == "Answer"]
        
        if not target_nodes:
            return []

        adj = {**{id: [] for id in info_node_map}, **{id: [] for id in action_node_map}}
        rev_adj = {**{id: [] for id in info_node_map}, **{id: [] for id in action_node_map}}
        for u, v in edges:
            if u in adj and v in adj:
                adj[u].append(v)
                rev_adj[v].append(u)

        queue = deque([start_node])
        distances = {node_id: float('inf') for node_id in adj}
        if start_node in distances:
            distances[start_node] = 0
        predecessors = {start_node: None}

        while queue:
            current_node = queue.popleft()
            for neighbor in adj.get(current_node, []):
                is_action = neighbor in action_node_map
                new_distance = distances[current_node] + (1 if is_action else 0)
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    queue.append(neighbor)
        
        best_target = None
        min_distance = float('inf')
        for target in target_nodes:
            if target in distances and distances[target] < min_distance:
                min_distance = distances[target]
                best_target = target

        if best_target is None:
            return []

        required_nodes = set()
        processing_queue = deque([best_target])
        
        while processing_queue:
            node = processing_queue.popleft()
            if node in required_nodes:
                continue
            required_nodes.add(node)
            
            if node in action_node_map:
                for pred in rev_adj.get(node, []):
                    if pred not in required_nodes:
                        processing_queue.append(pred)
            elif node in info_node_map:
                path_pred = predecessors.get(node)
                if path_pred and path_pred not in required_nodes:
                    processing_queue.append(path_pred)

        minimal_action_nodes = sorted([node for node in required_nodes if node in action_node_map])
        return minimal_action_nodes
    
    except Exception as e:
        raise e

def process_single_item(res_item: dict) -> dict | None:
    try:
        raw_message = res_item.get("raw_message")
        state_graph_str = res_item.get("state_graph_result")

        if not raw_message or not state_graph_str:
            return None
        
        graph_data = json.loads(state_graph_str)
        
        all_action_nodes_raw = graph_data.get("Action Node", [])
        all_action_node_ids = {list(node.keys())[0] for node in all_action_nodes_raw}

        assistant_message_count = 0
        if isinstance(raw_message, list):
            assistant_message_count = sum(
                1 for msg in raw_message 
                if isinstance(msg, dict) and msg.get("role", "").startswith("assistant")
            )
        
        if len(all_action_node_ids) != assistant_message_count:
            return None

        necessary_nodes_list = find_minimal_action_subgraph(graph_data)
        necessary_nodes_set = set(necessary_nodes_list)

        pruned_nodes_set = all_action_node_ids - necessary_nodes_set
        pruned_nodes_list = sorted(list(pruned_nodes_set))

        necessary_message = []
        pruned_message = []

        if isinstance(raw_message, list):
            for message in raw_message:
                role = message.get("role", "")
                if role.startswith("assistant"):
                    match = re.search(r'(\d+)$', role)
                    if match:
                        assistant_number = match.group(1)
                        current_action_id = f"A{assistant_number}"
                        
                        if current_action_id in necessary_nodes_set:
                            necessary_message.append(message)
                        elif current_action_id in pruned_nodes_set:
                            pruned_message.append(message)
        
        return {
            "raw_message": raw_message,
            "state_graph_result": state_graph_str,
            "necessary_nodes": necessary_nodes_list,
            "pruned_nodes": pruned_nodes_list,
            "necessary_message": necessary_message,
            "pruned_message": pruned_message,
        }
    except Exception:
        return None

def process_with_majority_vote(input_file_path: str):
    print(f"Starting vote processing file: {input_file_path}")
    print("Step 1: Reading and grouping all data...")

    query_groups = {}
    total_lines_read = 0
    
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            total_lines_read += 1
            try:
                item = json.loads(line)
                # 注意：这里假定 raw_message[1] 是首个 user；如不稳定需自己调整
                query_key = item['raw_message'][1]['content']
                if query_key not in query_groups:
                    query_groups[query_key] = []
                query_groups[query_key].append(item)
            except (json.JSONDecodeError, IndexError, KeyError):
                print(f"Warning: Line {i+1} format incorrect or missing key fields, skipped.")

    print(f"Step 1 completed: Read {total_lines_read} lines, formed {len(query_groups)} unique Query groups.")
    print("-" * 30)
    print("Step 2: Performing majority vote processing for each group...")

    processed_results = []
    processed_group_count = 0
    skipped_group_count = 0
    
    for query, items in query_groups.items():
        if len(items) != 3:
            print(f"Info: Query '{query[:50]}...' has {len(items)} data entries, not equal to 3, skipped.")
            skipped_group_count += 1
            continue

        candidates = []
        pruned_lists_for_vote = []
        for item in items:
            processed_result = process_single_item(item)
            if processed_result:
                candidates.append(processed_result)
                pruned_lists_for_vote.append(tuple(processed_result['pruned_nodes']))

        if len(candidates) < 2:
            skipped_group_count += 1
            continue
        
        vote_counts = Counter(pruned_lists_for_vote)
        winner_pruned_list_tuple = None
        
        for pruned_list, count in vote_counts.items():
            if count >= 2:
                winner_pruned_list_tuple = pruned_list
                break
        
        if winner_pruned_list_tuple is not None:
            winner_found = False
            for candidate in candidates:
                if tuple(candidate['pruned_nodes']) == winner_pruned_list_tuple:
                    processed_results.append(candidate)
                    processed_group_count += 1
                    winner_found = True
                    break
            if not winner_found:
                skipped_group_count += 1
        else:
            print(f"Info: Query '{query[:50]}...' valid results inconsistent, no majority vote, skipped.")
            skipped_group_count += 1

    print("-" * 30)
    print("Vote processing completed!")
    print(f"Successfully processed and retained Query groups: {processed_group_count}")
    print(f"Skipped Query groups due to inconsistency, insufficient data, or no majority vote: {skipped_group_count}")
    print(f"Total Query groups: {len(query_groups)}")
    print(f"Obtained {len(processed_results)} data entries after voting")
    # import pdb;pdb.set_trace()
    return processed_results

def format_context_with_system_user(messages: list, original_data: dict) -> str:
    formatted_messages = []
    
    if messages:
        for msg in messages:
            original_role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if original_role == 'system':
                continue
            if original_role.startswith('assistant'):
                display_role = 'assistant'
            else:
                display_role = original_role
                
            formatted_messages.append(f"{display_role}: {content}")
    
    if not formatted_messages:
        return "No previous context."
        
    return "\n".join(formatted_messages)

def format_messages_as_string(messages: list) -> str:
    formatted = []
    for msg in messages:
        role = msg.get('type', msg.get('role', 'unknown'))
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

def analyze_content_structure(content: str) -> dict:
    think_pattern = r'<think>(.*?)</think>'
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, content, re.DOTALL)
    tool_call_match = re.search(tool_call_pattern, content, re.DOTALL)
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    
    result = {
        'has_think': bool(think_match),
        'has_tool_call': bool(tool_call_match),
        'has_answer': bool(answer_match),
        'think_content': think_match.group(1).strip() if think_match else None,
        'tool_call_content': tool_call_match.group(1).strip() if tool_call_match else None,
        'answer_content': answer_match.group(1).strip() if answer_match else None
    }
    
    if result['has_think'] and result['has_tool_call']:
        result['type'] = 'tool_call'
    elif result['has_think'] and result['has_answer']:
        result['type'] = 'answer'
    else:
        result['type'] = 'unknown'
    
    return result

def validate_and_format_content(refined_content: str, original_structure: dict) -> str:
    if original_structure['type'] == 'tool_call':
        think_pattern = r'<refined_think>(.*?)</refined_think>'
        think_match = re.search(think_pattern, refined_content, re.DOTALL)
        
        if think_match:
            refined_think = think_match.group(1).strip()
            formatted_content = f"<think>\n{refined_think}\n</think>\n\n<tool_call>\n{original_structure['tool_call_content']}\n</tool_call>"
            return formatted_content
        else:
            return None
    
    elif original_structure['type'] == 'answer':
        think_pattern = r'<refined_think>(.*?)</refined_think>'
        answer_pattern = r'<refined_answer>(.*?)</refined_answer>'
        
        think_match = re.search(think_pattern, refined_content, re.DOTALL)
        answer_match = re.search(answer_pattern, refined_content, re.DOTALL)
        
        if think_match and answer_match:
            refined_think = think_match.group(1).strip()
            refined_answer = answer_match.group(1).strip()
            formatted_content = f"<think>\n{refined_think}\n</think>\n\n<answer>\n{refined_answer}\n</answer>"
            return formatted_content
        else:
            return None
    
    else:
        return refined_content

def load_ppl_model():
    global ppl_model, ppl_tokenizer, ppl_device
    
    print("Loading PPL model...")
    try:
        ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_PATH, trust_remote_code=True)
        ppl_model = AutoModelForCausalLM.from_pretrained(
            PPL_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True
        )
        
        if torch.cuda.device_count() > 1:
            print(f"  [PPL] Model loaded onto {torch.cuda.device_count()} GPUs (balanced distribution)")
        else:
            print(f"  [PPL] Model loaded onto single GPU")
        
        ppl_model.eval()
        
        if not hasattr(ppl_tokenizer, 'chat_template') or ppl_tokenizer.chat_template is None:
            print("  [PPL] Warning: Model has no chat template, using default format")
        
    except Exception as e:
        print(f"Failed to load PPL model: {e}")
        ppl_model = None
        ppl_tokenizer = None

def find_substring_token_indices(tokenizer, full_text, substring):
    substring_start_char = full_text.find(substring)
    if substring_start_char == -1:
        if full_text.find(substring.strip()) != -1:
            substring = substring.strip()
            substring_start_char = full_text.find(substring)
        else:
            raise ValueError("Substring not found in the full text even after stripping whitespace.")
            
    substring_end_char = substring_start_char + len(substring)

    full_encoding = tokenizer(full_text, return_offsets_mapping=True)
    offsets = full_encoding.offset_mapping
    
    start_token_idx = None
    end_token_idx = None
    
    for i, (start, end) in enumerate(offsets):
        if start_token_idx is None and start >= substring_start_char:
            start_token_idx = i
        
        if start < substring_end_char:
            end_token_idx = i

    if start_token_idx is not None and start_token_idx > 0 and offsets[start_token_idx][0] > substring_start_char:
        start_token_idx -= 1

    if start_token_idx is None or end_token_idx is None:
        raise ValueError("Couldn't map substring to tokens precisely.")
    
    return start_token_idx, end_token_idx

def calculate_substring_perplexity(model, tokenizer, full_text, substring, device="cuda"):
    try:
        start_token_idx, end_token_idx = find_substring_token_indices(tokenizer, full_text, substring)
    except ValueError:
        return float('inf')

    full_input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    
    if hasattr(model.config, 'max_position_embeddings') and full_input_ids.shape[1] > model.config.max_position_embeddings:
        print(f"  [PPL] Warning: Input length ({full_input_ids.shape[1]}) exceeds model's max length ({model.config.max_position_embeddings}). Skipping.")
        return float('inf')
        
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits
    
    if start_token_idx == 0:
        return float('inf')

    substring_logits = logits[0, start_token_idx - 1 : end_token_idx, :]
    substring_token_ids = full_input_ids[0, start_token_idx : end_token_idx + 1]
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(substring_logits, substring_token_ids)
    
    ppl = torch.exp(loss.mean()).item()
    
    return ppl

def calculate_ppl_for_candidate(previous_messages: list, candidate_text: str) -> float:
    global ppl_model, ppl_tokenizer, ppl_device
    
    if ppl_model is None or ppl_tokenizer is None:
        print("  [PPL] Warning: PPL model not loaded, cannot calculate PPL")
        return float('inf')
    
    try:
        full_messages = previous_messages.copy()
        full_messages.append({"role": "assistant", "content": candidate_text})
        
        try:
            full_text = ppl_tokenizer.apply_chat_template(
                full_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"  [PPL] Warning: Failed to use chat template ({e}), using default format")
            full_text = ""
            for msg in full_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if role == 'system':
                    full_text += f"System: {content}\n"
                elif role == 'user':
                    full_text += f"User: {content}\n"
                elif role == 'assistant':
                    full_text += f"Assistant: {content}\n"
        
        ppl = calculate_substring_perplexity(ppl_model, ppl_tokenizer, full_text, candidate_text, ppl_device)
        
        return ppl
    except Exception as e:
        print(f"  [PPL] Error calculating PPL: {e}")
        return float('inf')

def generate_single_candidate(client: OpenAI, user_message: str, structure: dict, candidate_id: int) -> str:
    format_retry_count = 0
    while format_retry_count < MAX_FORMAT_RETRIES:
        api_retry_count = 0
        while api_retry_count < MAX_API_RETRIES:
            try:
                messages = [{"role": "user", "content": INSTRUCTION + '\n' + user_message}]
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
                refined_content_llm = response.choices[0].message.content.strip()
                
                formatted_content = validate_and_format_content(refined_content_llm, structure)
                
                if formatted_content:
                    return formatted_content
                else:
                    print(f"  [Candidate {candidate_id}] Format retry {format_retry_count + 1}: Format does not meet requirements")
                    break
                    
            except Exception as e:
                api_retry_count += 1
                print(f"  [Candidate {candidate_id}] Format retry {format_retry_count + 1}, API call {api_retry_count} failed: {e}")
                if api_retry_count < MAX_API_RETRIES:
                    time.sleep(3)
                else:
                    print(f"  [Candidate {candidate_id}] Format retry {format_retry_count + 1} reached max API retry count")
        
        if api_retry_count >= MAX_API_RETRIES:
            print(f"  [Candidate {candidate_id}] API calls still failed after {format_retry_count + 1} format retries, candidate generation failed")
            return None
        
        format_retry_count += 1
        if format_retry_count < MAX_FORMAT_RETRIES:
            time.sleep(1)
    
    print(f"  [Candidate {candidate_id}] Reached max format retry count ({MAX_FORMAT_RETRIES}), generation failed")
    return None

def refine_content_with_llm(client: OpenAI, content_to_refine: str, previous_context: str, 
                           skipped_messages: list, structure: dict, previous_messages: list):
    if structure['type'] == 'unknown':
        print("  [!] Content structure unknown, skipping rewrite")
        return content_to_refine, None
    
    skipped_messages_str = format_messages_as_string(skipped_messages) if skipped_messages else "No skipped messages."
    
    if structure['type'] == 'tool_call':
        user_message_template = f"""
## Dialogue History (Already Stored Necessary Messages):
{previous_context}

## Skipped Messages (Between Previous Necessary Message and Current One):
{skipped_messages_str}

## Current Action to Refine (Thinking Process Only):
{content_to_refine}

## Important Instructions:
1. The skipped messages have been identified as providing no information gain. Merge the thinking process seamlessly.
2. Maintain the original first-person thinking style with self-questioning and logical reasoning.
3. Ensure perfect coherence between the previous necessary message's tool calls and the current one.
4. If skipped messages include failed tool attempts, incorporate that reasoning into the refined think.
5. Do not introduce any content not present in Dialogue History.

You are only allowed to refine the <think> section. 
Your output should be the refined <think> section only, without the <tool_call>.

## Output Format:
<refined_think>
YOUR REFINED THINK
</refined_think>

## Your Response:
"""
    elif structure['type'] == 'answer':
        user_message_template = f"""
## Dialogue History (Already Stored Necessary Messages):
{previous_context}

## Skipped Messages (Between Previous Necessary Message and Current One):
{skipped_messages_str}

## Current Action to Refine (Thinking Process and Answer):
{content_to_refine}

## Important Instructions:
1. The skipped messages have been identified as providing no information gain. Merge the thinking process seamlessly.
2. Maintain the original first-person thinking style with self-questioning and logical reasoning.
3. Ensure perfect coherence between the previous necessary message and the current answer.
4. If skipped messages include failed tool attempts, incorporate that reasoning into the refined think.
5. Do not introduce any content not present in Dialogue History.
6. First, carefully identify the **core answer** the assistant is trying to give (the key conclusion / final result that directly responds to the user's question). This core answer must be preserved in meaning.
7. Remove any explanation, detail, or statement that is **not shown by the Dialogue History**. In other words, delete all hallucinated or extraneous information that does not appear in the Dialogue History and is not logically required to state the core answer.

You should refine both the <think> and <answer> sections.

## Output Format:
<refined_think>
YOUR REFINED THINK
</refined_think>

<refined_answer>
YOUR REFINED ANSWER
</refined_answer>

## Your Response:
"""
    
    print(f"  [Concurrent] Starting concurrent generation of {NUM_CANDIDATES} candidates...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CANDIDATES) as executor:
        futures = []
        
        for i in range(NUM_CANDIDATES):
            future = executor.submit(
                generate_single_candidate,
                client,
                user_message_template,
                structure,
                i+1
            )
            futures.append(future)
        
        candidates = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            candidate = future.result()
            if candidate:
                candidates.append(candidate)
    
    if not candidates:
        print(f"  [!] Unable to generate any candidates, rewrite failed")
        return None, None
    
    print(f"  [Concurrent] Successfully generated {len(candidates)} candidates")
    
    if len(candidates) > 1:
        print(f"  [PPL] Starting to select best result from {len(candidates)} candidates...")
        
        candidate_ppls = []
        for i, candidate in enumerate(candidates):
            ppl = calculate_ppl_for_candidate(previous_messages, candidate)
            candidate_ppls.append((ppl, candidate))
            print(f"  [PPL] Candidate {i+1} P: {ppl:.4f}")
        
        best_ppl, best_candidate = min(candidate_ppls, key=lambda x: x[0])
        print(f"  [PPL] Selected candidate with lowest PPL (PPL: {best_ppl:.4f})")
        
        return best_candidate, best_ppl
    else:
        ppl = calculate_ppl_for_candidate(previous_messages, candidates[0])
        print(f"  [PPL] Single candidate PPL: {ppl:.4f}")
        return candidates[0], ppl

def refine_single_data(data, client):
    try:
        if "raw_message" not in data or not data["raw_message"]:
            return data, "no_raw_message"
        
        raw_messages = data["raw_message"]
        necessary_nodes = data.get("necessary_nodes", [])
        
        refined_ppl_values = []
        has_refined = False
        refine_failed = False
        
        if not necessary_nodes:
            messages = []
            for msg in raw_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system" or role == "user":
                    messages.append({"role": role, "content": content})
            
            data["messages"] = messages
            data["max_ppl"] = 1.0
            data["avg_ppl"] = 1.0
            return data, "no_necessary_nodes"
        
        necessary_nums = []
        for node in necessary_nodes:
            if node.startswith('A'):
                try:
                    num = int(node[1:])
                    necessary_nums.append(num)
                except ValueError:
                    continue
        
        necessary_nums.sort()
        
        if not necessary_nums:
            data["messages"] = []
            data["max_ppl"] = 1.0
            data["avg_ppl"] = 1.0
            return data, "no_valid_necessary_nodes"
        
        system_message = None
        all_messages = []
        assistant_counter = 0
        assistant_num_to_index = {}
        
        first_assistant_num_in_raw = None
        
        for msg in raw_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = {"role": "system", "content": content}
            elif role.startswith("assistant"):
                assistant_counter += 1
                if first_assistant_num_in_raw is None:
                    first_assistant_num_in_raw = assistant_counter
                
                msg_info = {
                    "type": "assistant",
                    "original_role": role,
                    "content": content,
                    "assistant_num": assistant_counter,
                    "is_necessary": assistant_counter in necessary_nums,
                    "index": len(all_messages),
                    "is_first_in_raw": assistant_counter == first_assistant_num_in_raw
                }
                all_messages.append(msg_info)
                assistant_num_to_index[assistant_counter] = len(all_messages) - 1
            else:
                all_messages.append({
                    "type": role,
                    "original_role": role,
                    "content": content,
                    "index": len(all_messages)
                })
        
        messages = []
        
        if system_message:
            messages.append(system_message)
        
        first_user_index = -1
        for i, msg in enumerate(all_messages):
            if msg["type"] == "user":
                first_user_index = i
                break
        
        if first_user_index == -1:
            return data, "no_user_message_found"
        
        messages.append({
            "role": "user",
            "content": all_messages[first_user_index]["content"]
        })
        
        necessary_assistant_positions = []
        for num in necessary_nums:
            if num in assistant_num_to_index:
                necessary_assistant_positions.append(assistant_num_to_index[num])
        
        if not necessary_assistant_positions:
            data["messages"] = messages
            data["max_ppl"] = 1.0
            data["avg_ppl"] = 1.0
            return data, "no_necessary_assistant_found"
        
        print(f"  [DEBUG] necessary_nums: {necessary_nums}")
        print(f"  [DEBUG] necessary_assistant_positions: {necessary_assistant_positions}")
        
        refined_assistant_contents = {}
        
        for pos_idx, assistant_pos in enumerate(necessary_assistant_positions):
            assistant_msg = all_messages[assistant_pos]
            content_to_refine = assistant_msg["content"]
            
            structure = analyze_content_structure(content_to_refine)
            
            if pos_idx == 0:
                is_first_in_raw = assistant_msg.get("is_first_in_raw", False)
                
                if is_first_in_raw:
                    refined_content = content_to_refine
                    
                    messages.append({
                        "role": "assistant",
                        "content": refined_content
                    })
                    
                    next_msg_idx = assistant_pos + 1
                    if pos_idx < len(necessary_assistant_positions) - 1:
                        if next_msg_idx < len(all_messages) and all_messages[next_msg_idx]["type"] == "user":
                            messages.append({
                                "role": "user",
                                "content": all_messages[next_msg_idx]["content"]
                            })
                else:
                    skipped_messages = []
                    for i in range(first_user_index + 1, assistant_pos):
                        skipped_messages.append(all_messages[i])
                    
                    previous_context = []
                    for msg in messages:
                        previous_context.append({"role": msg["role"], "content": msg["content"]})
                    
                    previous_context_str = format_context_with_system_user(previous_context, data)
                    
                    refined_content, ppl_value = refine_content_with_llm(
                        client, content_to_refine, previous_context_str, skipped_messages, structure, messages
                    )
                    
                    if refined_content is None:
                        refine_failed = True
                        break
                    
                    if ppl_value is not None and ppl_value != float('inf'):
                        refined_ppl_values.append(ppl_value)
                        has_refined = True
                    
                    refined_assistant_contents[assistant_pos] = refined_content
                    
                    messages.append({
                        "role": "assistant",
                        "content": refined_content
                    })
                    
                    next_msg_idx = assistant_pos + 1
                    if pos_idx < len(necessary_assistant_positions) - 1:
                        if next_msg_idx < len(all_messages) and all_messages[next_msg_idx]["type"] == "user":
                            messages.append({
                                "role": "user",
                                "content": all_messages[next_msg_idx]["content"]
                            })
            
            else:
                prev_assistant_pos = necessary_assistant_positions[pos_idx - 1]
                
                is_adjacent = (assistant_pos - prev_assistant_pos) == 2
                
                if is_adjacent:
                    if structure['type'] == 'answer':
                        skipped_messages = []
                        for i in range(prev_assistant_pos + 1, assistant_pos):
                            skipped_messages.append(all_messages[i])
                        
                        previous_context = []
                        for msg in messages:
                            previous_context.append({"role": msg["role"], "content": msg["content"]})
                        
                        previous_context_str = format_context_with_system_user(previous_context, data)
                        
                        refined_content, ppl_value = refine_content_with_llm(
                            client, content_to_refine, previous_context_str, skipped_messages, structure, messages
                        )
                        
                        if refined_content is None:
                            refine_failed = True
                            break
                        
                        if ppl_value is not None and ppl_value != float('inf'):
                            refined_ppl_values.append(ppl_value)
                            has_refined = True
                        
                        refined_assistant_contents[assistant_pos] = refined_content
                        
                        messages.append({
                            "role": "assistant",
                            "content": refined_content
                        })
                        
                        next_msg_idx = assistant_pos + 1
                        if pos_idx < len(necessary_assistant_positions) - 1:
                            if next_msg_idx < len(all_messages) and all_messages[next_msg_idx]["type"] == "user":
                                messages.append({
                                    "role": "user",
                                    "content": all_messages[next_msg_idx]["content"]
                                })
                    else:
                        refined_content = content_to_refine
                        
                        messages.append({
                            "role": "assistant",
                            "content": refined_content
                        })
                        
                        next_msg_idx = assistant_pos + 1
                        if pos_idx < len(necessary_assistant_positions) - 1:
                            if next_msg_idx < len(all_messages) and all_messages[next_msg_idx]["type"] == "user":
                                messages.append({
                                    "role": "user",
                                    "content": all_messages[next_msg_idx]["content"]
                                })
                else:
                    skipped_messages = []
                    for i in range(prev_assistant_pos + 1, assistant_pos):
                        skipped_messages.append(all_messages[i])
                    
                    previous_context = []
                    for msg in messages:
                        previous_context.append({"role": msg["role"], "content": msg["content"]})
                    
                    previous_context_str = format_context_with_system_user(previous_context, data)
                    
                    refined_content, ppl_value = refine_content_with_llm(
                        client, content_to_refine, previous_context_str, skipped_messages, structure, messages
                    )
                    
                    if refined_content is None:
                        refine_failed = True
                        break
                    
                    if ppl_value is not None and ppl_value != float('inf'):
                        refined_ppl_values.append(ppl_value)
                        has_refined = True
                    
                    refined_assistant_contents[assistant_pos] = refined_content
                    
                    messages.append({
                        "role": "assistant",
                        "content": refined_content
                    })
                    
                    next_msg_idx = assistant_pos + 1
                    if pos_idx < len(necessary_assistant_positions) - 1:
                        if next_msg_idx < len(all_messages) and all_messages[next_msg_idx]["type"] == "user":
                            messages.append({
                                "role": "user",
                                "content": all_messages[next_msg_idx]["content"]
                            })
        
        if refine_failed:
            return data, "refine_failed"
        
        data["messages"] = messages
        
        if not has_refined:
            data["max_ppl"] = 1.0
            data["avg_ppl"] = 1.0
        else:
            if refined_ppl_values:
                data["max_ppl"] = max(refined_ppl_values)
                data["avg_ppl"] = sum(refined_ppl_values) / len(refined_ppl_values)
                data["refined_ppl_values"] = refined_ppl_values
            else:
                data["max_ppl"] = 1.0
                data["avg_ppl"] = 1.0
        
        if refined_assistant_contents:
            data["refined_assistant_contents"] = {
                str(pos): refined_assistant_contents[pos] 
                for pos in refined_assistant_contents
            }
        
        return data, "success"
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return data, f"error: {str(e)}"

def refine_data_list(data_list):
    print(f"Starting rewrite processing for {len(data_list)} data entries...")
    print(f"Using model: {MODEL_NAME}")
    print(f"Parallel worker threads: {MAX_WORKERS}")
    print(f"PPL selection model: {PPL_MODEL_PATH}")
    print(f"Generating {NUM_CANDIDATES} candidates per rewrite (concurrent generation)")
    print(f"API call retry count: {MAX_API_RETRIES} times, wait 3 seconds each")
    print(f"Format retry count: {MAX_FORMAT_RETRIES} times, wait 1 second each")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        clients = [OpenAI(api_key=REWRITER_API_KEY, base_url=REWRITER_BASE_URL) for _ in range(MAX_WORKERS)]
        
        process_funcs = [partial(refine_single_data, client=client) for client in clients]
        
        futures = []
        for i, data in enumerate(data_list):
            process_func = process_funcs[i % len(process_funcs)]
            future = executor.submit(process_func, data)
            futures.append((i, future))
        
        results = []
        for i, future in tqdm(futures, desc="Processing data", total=len(futures)):
            try:
                result_data, status = future.result()
                results.append((i, result_data, status))
            except Exception as e:
                print(f"Error processing data: {e}")
                if i < len(data_list):
                    results.append((i, data_list[i], f"future_error: {str(e)}"))
    
    results.sort(key=lambda x: x[0])
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="An end-to-end pipeline that performs majority vote selection on state graph results and rewrites thought."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL File Path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL File Path"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input
    output_file = args.output

    print("=" * 60)
    print("Starting end-to-end processing pipeline")
    print("=" * 60)
    
    print("\nPhase 1: Loading PPL model...")
    load_ppl_model()
    
    print("\n" + "=" * 60)
    print("Phase 2: Performing majority vote processing...")
    print("=" * 60)
    voted_results = process_with_majority_vote(input_file)
    
    if not voted_results:
        print("Error: No valid data obtained after vote processing, program terminated.")
        return
    
    print("\n" + "=" * 60)
    print("Phase 3: Performing message rewrite processing...")
    print("=" * 60)
    refined_results = refine_data_list(voted_results)
    
    print("\n" + "=" * 60)
    print("Phase 4: Writing final result file...")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    refine_failed_count = 0
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for _, result_data, status in refined_results:
            if status == "success":
                outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                success_count += 1
            elif status == "refine_failed":
                refine_failed_count += 1
            elif status in ["no_necessary_message", "unknown_structure", "no_necessary_nodes", 
                           "no_valid_necessary_nodes", "no_user_message_found", "no_necessary_assistant_found"]:
                skip_count += 1
            else:
                error_count += 1
    
    print(f"\nProcessing statistics:")
    print(f"  Successfully processed: {success_count} entries")
    print(f"  Rewrite failed and discarded: {refine_failed_count} entries")
    print(f"  Skipped processing: {skip_count} entries")
    print(f"  Processing errors: {error_count} entries")
    print(f"  Total input: {len(voted_results)} entries")
    
    print("\n" + "=" * 60)
    print("End-to-end processing completed!")
    print(f"Final result saved to: '{output_file}'")
    print(f"Output file contains {success_count} successfully processed data entries")
    print("=" * 60)

if __name__ == "__main__":
    main()
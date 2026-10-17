STAGE_1_PROMPT = """You are given {num_clips} **shuffled** video clips (**containing audio**) that were created by slicing one original video into {num_clips} equal-length temporal segments.

Here are the clips, each tagged with an index reflecting the current (shuffled) order in which they are shown. Please pay attention to both the visual actions and the continuity of the audio (e.g., speech sentences, background music, or sound events).

{clip_placeholders}

Task:
1. Analyze the visual content and listen to the audio tracks.
2. Mentally reassemble the original video by arranging the clips in their **correct chronological order** (earliest segment first, latest segment last). Use **audio continuity** as a key clue.
3. Output the clip indices in that order, separated by commas.

Answer format example:
2, 3, 1, 4, 6, 5
You **FIRST** think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""

STAGE_2_PROMPT = """You have just completed a video temporal reordering task based on this video to understand its timeline. Here is the summary of your previous analysis:

<stage1_recap>
**Task**: Reassemble shuffled video clips into chronological order.
**Your Thinking Process**: 
{stage1_thinking}

**Your Predicted Order**: {stage1_prediction}
**The Ground Truth (Correct) Order**: {stage1_gt}
</stage1_recap>

<reflection_instruction>
Please compare your predicted order with the Ground Truth. 
1. If your prediction was correct, reinforce your understanding of the timeline.
2. If your prediction was incorrect, critically analyze the root causes of the discrepancy. Recalibrate the **temporal coherence, causal relationships, and cross-modal alignment (visual and audio)** to rectify your understanding of the temporal dynamics.
Use the Correct Order to mentally reconstruct the true timeline of events and causal relationships in the video.
</reflection_instruction>

Now, based on your refined understanding of the complete video content (visual + audio) and the correct timeline, please answer the following question.

<task>
**Question**: {question}
**Candidates**:
{candidates_str}
</task>

You **FIRST** output your reflection comparing your predicted order with the Ground Truth and analyzing the root causes of any discrepancies, **THEN** think about the reasoning process for the current question incorporating your reflection, and **FINALLY** provide the final answer. The reflection MUST BE enclosed within <reflection> </reflection> tags. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer (just the option letter) MUST BE enclosed within <answer> </answer> tags."""

STANDARD_PROMPT = """Please answer the following question based on the complete video content (visual + audio).

<task>
**Question**: {question}
**Candidates**:
{candidates_str}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer (just the option letter) MUST BE enclosed within <answer> </answer> tags."""

GENERATION_PROMPT = """Please answer the following question based on the complete video content (visual + audio).

<task>
**Question**: {question}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""

AUDIO_INFER_PROMPT = """Please answer the following question based on the complete audio content.

<task>
**Question**: {question}
**Candidates**:
{candidates_str}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer (just the option letter) MUST BE enclosed within <answer> </answer> tags."""

AUDIO_GENERATION_PROMPT = """Please answer the following question based on the complete audio content.

<task>
**Question**: {question}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""

VIDEO_INFER_PROMPT = """Please answer the following question based on the complete video content.

<task>
**Question**: {question}
**Candidates**:
{candidates_str}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer (just the option letter) MUST BE enclosed within <answer> </answer> tags."""

VIDEO_GENERATION_PROMPT = """Please answer the following question based on the complete video content.

<task>
**Question**: {question}
</task>

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""

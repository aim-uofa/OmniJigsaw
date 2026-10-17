import logging
from typing import Optional

import torch

# Configure module logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_feat_extract_output_lengths(input_lengths):
    """Return audio-encoder token lengths after convolutional downsampling."""

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[torch.Tensor],
    grid_hs: list[torch.Tensor],
    grid_ws: list[torch.Tensor],
):
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_rope_index(
    processor,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen3-Omni temporal, height, and width position IDs.

    Vision positions use the processor's spatial merge factor and a fixed
    temporal scale of 13 positions per second. Embedded audio positions are
    interleaved with video positions on the same timeline. Text-only positions
    are shared across all three axes.

    Args:
        processor: Qwen3-Omni processor providing token IDs and spatial settings.
        input_ids: Token IDs with shape (batch_size, sequence_length).
        image_grid_thw: Optional image grids with shape (num_images, 3).
        video_grid_thw: Optional video grids with shape (num_videos, 3).
        attention_mask: Token mask; one marks valid tokens and zero marks padding.
        use_audio_in_video: Whether video tokens include an embedded audio stream.
        audio_seqlens: Audio feature lengths before encoder downsampling.
        second_per_grids: Seconds per temporal grid for each video.

    Returns:
        A floating-point tensor with shape (batch_size, 3, sequence_length).
        Position deltas are not returned by this project adapter.
    """

    spatial_merge_size = processor.image_processor.merge_size

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.vision_bos_token)
    audio_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_bos_token)

    # TODO: Read the temporal position scale from the model configuration.
    position_id_per_seconds = 13.0

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask == 1
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i]]
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                    remain_videos > 0 or remain_images > 0
                ):
                    ed_vision_start = input_tokens.index(vision_start_token_id, st)
                else:
                    ed_vision_start = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio_start = input_tokens.index(audio_start_token_id, st)
                else:
                    ed_audio_start = len(input_tokens) + 1
                min_ed = min(ed_vision_start, ed_audio_start)

                text_len = min_ed - st
                if text_len != 0:
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += text_len
                # Embedded audio-video inputs include two boundary tokens.
                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    bos_len, eos_len = 2, 2
                else:
                    bos_len, eos_len = 1, 1
                llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                st_idx += bos_len
                # Assign sequential positions to audio tokens.
                if min_ed == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + audio_len + eos_len)
                    audio_idx += 1
                    remain_audios -= 1

                # Assign temporal and spatial positions to image tokens.
                elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + image_len + eos_len)
                    image_idx += 1
                    remain_images -= 1

                # Scale video temporal positions by the grid interval.
                elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == video_token_id:
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).float()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += int(text_len + bos_len + video_len + eos_len)
                    video_idx += 1
                    remain_videos -= 1

                # Interleave audio and video positions on a shared timeline.
                elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).float()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_data_index, audio_data_index = 0, 0
                    while (
                        video_data_index < video_llm_pos_ids.shape[-1]
                        and audio_data_index < audio_llm_pos_ids.shape[-1]
                    ):
                        if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]])

                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]])

                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                    st += int(text_len + bos_len + audio_len + video_len + eos_len)

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids.permute(1, 0, 2)
    else:
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids.permute(1, 0, 2)

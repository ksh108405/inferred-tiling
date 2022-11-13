import numpy as np


def frame_match_bboxes(prev_frame_bboxes, prev_frame_labels, cur_frame_bboxes,
                       cur_frame_labels, trans_thresh=0.1, video_w=1920, video_h=1080):
    """
    Match bboxes from previous frame to current frame.
    """
    result = np.ones(cur_frame_labels.size) * -1
    for prev_idx, prev_frame_bbox in enumerate(prev_frame_bboxes):
        matched_bbox_idx = []  # list of bbox indices in current frame that match the previous frame bbox
        # 1. label match
        filter_1 = []
        for bbox_idx, label in enumerate(cur_frame_labels):
            if bbox_idx in matched_bbox_idx:  # already used
                continue
            if prev_frame_labels[prev_idx] == label:
                filter_1.append(bbox_idx)

        if len(filter_1) > 0:
            # 2. bbox center point distance match
            cx_prev = (prev_frame_bbox[0] + prev_frame_bbox[2]) / 2
            cy_prev = (prev_frame_bbox[1] + prev_frame_bbox[3]) / 2

            # 2-1. filter by translation threshold
            filter_2_1 = []
            filter_2_1_dist = []
            for bbox_idx in filter_1:
                bbox = cur_frame_bboxes[bbox_idx]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if abs(cx - cx_prev) < (trans_thresh * video_w) and abs(cy - cy_prev) < (trans_thresh * video_h):
                    filter_2_1.append(bbox_idx)
                    filter_2_1_dist.append((cx - cx_prev) ** 2 + (cy - cy_prev) ** 2)

            if len(filter_2_1) > 0:
                # 2-2. choose minimal distance bbox
                filter_2_1_dist = np.array(filter_2_1_dist)
                matched_bbox = filter_2_1[np.argmin(filter_2_1_dist)]
                # match bbox
                matched_bbox_idx.append(matched_bbox)
                result[matched_bbox] = prev_idx
                continue

    return result

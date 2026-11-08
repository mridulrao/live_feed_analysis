"""Compare independently localized targets with CSRT boxes from the same frame."""


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0


def tracks_match(tracks, detections, minimum_iou):
    """Every tracked object needs a distinct matching detection; labels are not IDs."""
    if not tracks or len(detections) < len(tracks):
        return False
    candidates = [[i for i, detection in enumerate(detections)
                   if intersection_over_union(track.box, detection.box) >= minimum_iou] for track in tracks]
    assigned = {}
    def assign(track_index, seen):
        for detection_index in candidates[track_index]:
            if detection_index in seen:
                continue
            seen.add(detection_index)
            if detection_index not in assigned or assign(assigned[detection_index], seen):
                assigned[detection_index] = track_index
                return True
        return False
    return all(assign(i, set()) for i in range(len(tracks)))

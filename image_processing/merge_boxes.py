from image_processing.utils import get_center, merge_bounding_boxes

def cluster_date_of_expire(bboxes, horizontal_threshold=100):
    horizontally_sorted = sorted(bboxes, key=lambda b: (b[0], b[1]))
    date_of_expire_cluster = []

    for i, box in enumerate(horizontally_sorted):
        if not date_of_expire_cluster:
            date_of_expire_cluster.append(box)
            continue

        x_min = box[0]
        last_x_min = date_of_expire_cluster[-1][0]

        if abs(x_min - last_x_min) > horizontal_threshold:
            return date_of_expire_cluster, horizontally_sorted[i:]
        
        date_of_expire_cluster.append(box)
    
    return date_of_expire_cluster, []

def cluster_words(bboxes, vertical_threshold=15, horizontal_threshold=400):
    """Cluster text boxes into lines based on their vertical and horizontal positions"""
    vertically_sorted = sorted(bboxes, key=lambda b: (b[1], b[0]))
    word_clusters = []
    current_cluster = []

    for box in vertically_sorted:
        if not current_cluster:
            current_cluster.append(box)
            continue

        cluster_box = merge_bounding_boxes(current_cluster)
        vertical_distance = abs(get_center(box)[1] - get_center(cluster_box)[1])
        horizontal_distance = abs(get_center(box)[0] - get_center(cluster_box)[0])

        # Student ID cards have text fields that can span most of the width
        # and can have varying vertical spacing between lines
        # We'll be more lenient with vertical spacing
        if vertical_distance < vertical_threshold and horizontal_distance < horizontal_threshold:
            current_cluster.append(box)
        else:
            # Before starting a new cluster, sort boxes horizontally and merge
            current_cluster = sorted(current_cluster, key=lambda b: b[0])
            word_clusters.append(current_cluster)
            current_cluster = [box]
            
    if current_cluster:
        current_cluster = sorted(current_cluster, key=lambda b: b[0])
        word_clusters.append(current_cluster)
    
    return word_clusters

def merge_bboxes(bboxes):
    """Merge text boxes into coherent text lines"""
    # For student ID format, we cluster words vertically but with more tolerance
    # This helps handle cases where text might be slightly misaligned
    word_clusters = cluster_words(bboxes, vertical_threshold=20, horizontal_threshold=400)
    
    # Sort clusters vertically to maintain reading order
    word_clusters.sort(key=lambda cluster: min(box[1] for box in cluster))
    
    return [merge_bounding_boxes(cluster) for cluster in word_clusters if cluster]
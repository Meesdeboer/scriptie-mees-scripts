def count_overlapping_chars(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Both strings must have the same length")

    count = 0
    for char1, char2 in zip(str1, str2):
        if char1 == char2:
            count += 1

    return count

def match_license_plate_to_database(lp_prediction, lp_database):
    max_overlaps = 0
    max_overlap_lp = ''
    with open(lp_database, 'r') as f:
        for line in f:
            overlap_count = count_overlapping_chars(lp_prediction, line.strip())
            if overlap_count > max_overlaps:
                max_overlaps = overlap_count
                max_overlap_lp = line.strip()

    if max_overlaps < 5:
        return None, 0

    return max_overlap_lp, max_overlaps
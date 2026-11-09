"""
Phase 2: Character-level boundary aligner for Indian Legal NER.
Maps subword B-I-O predictions back to raw document character offsets.
"""
def align_subwords_to_character_spans(text, token_labels, offset_mapping):
    entities = []
    current_label = None
    current_start = None
    current_end = None

    for label, (start, end) in zip(token_labels, offset_mapping):
        if start == end:
            continue
        if label.startswith("B-"):
            if current_label:
                entities.append({"label": current_label, "start": current_start, "end": current_end, "text": text[current_start:current_end]})
            current_label = label[2:]
            current_start = start
            current_end = end
        elif label.startswith("I-") and current_label == label[2:]:
            current_end = end
        else:
            if current_label:
                entities.append({"label": current_label, "start": current_start, "end": current_end, "text": text[current_start:current_end]})
                current_label = None
    if current_label:
        entities.append({"label": current_label, "start": current_start, "end": current_end, "text": text[current_start:current_end]})
    return entities

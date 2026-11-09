"""
Character-level span reconstruction from subword B-I-O token predictions.
Resolves whitespace boundary offsets and tokenization fragmentation.
"""
def align_predictions_to_char_offsets(text, token_preds, offset_mapping):
    entities = []
    current_entity = None
    for pred_label, (start_char, end_char) in zip(token_preds, offset_mapping):
        if start_char == end_char:
            continue
        if pred_label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "label": pred_label[2:],
                "start": start_char,
                "end": end_char
            }
        elif pred_label.startswith("I-"):
            if current_entity and current_entity["label"] == pred_label[2:]:
                current_entity["end"] = end_char
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": pred_label[2:],
                    "start": start_char,
                    "end": end_char
                }
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities

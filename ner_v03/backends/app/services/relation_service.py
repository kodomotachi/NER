from app.services.ner_service import predict_entities


def _merge_adjacent_entities(sentence: str, entities: list[dict]):
    merged_entities = []

    for entity in entities:
        if not merged_entities or merged_entities[-1]["label"] != entity["label"]:
            merged_entities.append(entity.copy())
            continue

        previous_text = merged_entities[-1]["text"]
        current_text = entity["text"]
        joined_with_space = f"{previous_text} {current_text}"
        joined_without_space = f"{previous_text}{current_text}"

        if joined_with_space in sentence:
            merged_entities[-1]["text"] = joined_with_space
        elif joined_without_space in sentence:
            merged_entities[-1]["text"] = joined_without_space
        else:
            merged_entities[-1]["text"] = joined_with_space

    return merged_entities


def analyze_relationships(text: str):
    results = []

    sentences = [sentence.strip() for sentence in text.split(".")]
    for sentence in sentences:
        if not sentence:
            continue

        evidence = f"{sentence}."
        entities = _merge_adjacent_entities(evidence, predict_entities(evidence))

        people = [entity["text"] for entity in entities if entity["label"] == "PER"]
        organizations = [entity["text"] for entity in entities if entity["label"] == "ORG"]
        locations = [entity["text"] for entity in entities if entity["label"] == "LOC"]

        organization = organizations[0] if organizations else None
        location = locations[0] if locations else None

        for person in people:
            results.append(
                {
                    "person": person,
                    "organization": organization,
                    "location": location,
                    "evidence": evidence,
                }
            )

    return {
        "people_count": len(results),
        "results": results,
    }

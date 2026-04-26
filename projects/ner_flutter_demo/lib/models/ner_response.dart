class NerResponse {
  const NerResponse({required this.entities});

  factory NerResponse.fromJson(Map<String, dynamic> json) {
    final entitiesJson = json['entities'] as List<dynamic>? ?? [];

    return NerResponse(
      entities: entitiesJson
          .map((item) => NerEntity.fromJson(item as Map<String, dynamic>))
          .toList(),
    );
  }

  final List<NerEntity> entities;
}

class NerEntity {
  const NerEntity({
    required this.text,
    required this.label,
    required this.score,
    required this.start,
    required this.end,
  });

  factory NerEntity.fromJson(Map<String, dynamic> json) {
    return NerEntity(
      text: json['text'] as String? ?? '',
      label: json['label'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      start: (json['start'] as num?)?.toInt() ?? 0,
      end: (json['end'] as num?)?.toInt() ?? 0,
    );
  }

  final String text;
  final String label;
  final double score;
  final int start;
  final int end;
}

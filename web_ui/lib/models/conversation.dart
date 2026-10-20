import 'chat_message.dart';

/// Header the backend prepends to the trailing citation block of a streamed
/// answer (see `QAChain.query_stream`): `"\n\n---\nIzvori:\n[1] f.pdf ..."`.
const _sourcesHeader = '\n\n---\nIzvori:';

/// A named chat thread — the unit that is persisted to localStorage.
class Conversation {
  Conversation({
    required this.id,
    required this.title,
    List<ChatMessage>? messages,
    DateTime? createdAt,
    DateTime? updatedAt,
  })  : messages = messages ?? [],
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  final String id;
  String title;
  final List<ChatMessage> messages;
  final DateTime createdAt;
  DateTime updatedAt;

  /// Marks the conversation as changed just now.
  void touch() => updatedAt = DateTime.now();

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'createdAt': createdAt.toIso8601String(),
        'updatedAt': updatedAt.toIso8601String(),
        'messages': messages.map((m) => m.toJson()).toList(),
      };

  factory Conversation.fromJson(Map<String, dynamic> json) => Conversation(
        id: json['id'] as String? ?? '',
        title: json['title'] as String? ?? 'Novi razgovor',
        createdAt: DateTime.tryParse(json['createdAt'] as String? ?? ''),
        updatedAt: DateTime.tryParse(json['updatedAt'] as String? ?? ''),
        messages: (json['messages'] as List<dynamic>? ?? [])
            .whereType<Map<String, dynamic>>()
            .map(ChatMessage.fromJson)
            .toList(),
      );
}

/// Splits a streamed answer into the answer text and the citation lines the
/// backend appends after it:
///
///     Predmet nosi 6 ESPB [Izvor 1].
///
///     ---
///     Izvori:
///     [1] informatika_oas_2021.pdf — strana 12
///
/// Inline `[Izvor N]` markers are kept — they point at the numbered entries
/// of the returned list. Answers without a citation block are returned
/// unchanged (with no sources).
({String text, List<String> sources}) splitSources(String content) {
  final index = content.lastIndexOf(_sourcesHeader);
  if (index == -1) return (text: content.trim(), sources: const []);

  final block = content.substring(index + _sourcesHeader.length);
  final sources = [
    for (final line in block.split('\n'))
      if (line.trim().isNotEmpty) line.trim(),
  ];

  return (text: content.substring(0, index).trim(), sources: sources);
}

import 'chat_message.dart';

class Conversation {
  Conversation({
    required this.id,
    required this.title,
    List<ChatMessage>? messages,
    DateTime? createdAt,
    DateTime? updatedAt,
    this.serverSessionId,
    this.serverSessionToken,
    this.serverBaseUrl,
  })  : messages = messages ?? [],
        createdAt = createdAt ?? DateTime.now(),
        updatedAt = updatedAt ?? DateTime.now();

  final String id;
  String title;
  String? serverSessionId;
  String? serverSessionToken;
  String? serverBaseUrl;
  final List<ChatMessage> messages;
  final DateTime createdAt;
  DateTime updatedAt;

  void touch() => updatedAt = DateTime.now();

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'serverSessionId': serverSessionId,
        'serverSessionToken': serverSessionToken,
        'serverBaseUrl': serverBaseUrl,
        'createdAt': createdAt.toIso8601String(),
        'updatedAt': updatedAt.toIso8601String(),
        'messages': messages.map((m) => m.toJson()).toList(),
      };

  factory Conversation.fromJson(Map<String, dynamic> json) => Conversation(
        id: json['id'] as String,
        title: json['title'] as String,
        serverSessionId: json['serverSessionId'] as String?,
        serverSessionToken: json['serverSessionToken'] as String?,
        serverBaseUrl: json['serverBaseUrl'] as String?,
        createdAt:
            DateTime.tryParse(json['createdAt'] as String? ?? '') ??
                DateTime.now(),
        updatedAt:
            DateTime.tryParse(json['updatedAt'] as String? ?? '') ??
                DateTime.now(),
        messages: (json['messages'] as List<dynamic>? ?? [])
            .map((m) => ChatMessage.fromJson(m as Map<String, dynamic>))
            .toList(),
      );
}

final _citationMarkerRe =
    RegExp(r'\[\s*izvor\b[^\]]*\]', caseSensitive: false);

/// Removes any `[Izvor N]` markers that may still be present in an answer.
///
/// The backend strips them from the answer text and sends the citation
/// numbers separately in the `sources` event; this is a safety net for
/// persisted messages written by an older backend.
String stripCitationMarkers(String text) {
  var cleaned = text.replaceAll(_citationMarkerRe, '');
  cleaned = cleaned.replaceAllMapped(
    RegExp(r'[ \t]+([,.;:!?])'),
    (match) => match.group(1)!,
  );
  return cleaned;
}

/// Splits an assistant answer into the main text and a list of sources.
///
/// Primary source of truth is [ChatMessage.sources] (populated from the
/// structured `sources` SSE event).  This parser is kept as a fallback for
/// answers persisted before that event existed, i.e. answers that end with
/// a `--- Izvori:` block:
///
///     --- Izvori:
///     [1] informatika_oas_2021.pdf — strana 12
///
/// In every case the returned text is free of inline `[Izvor N]` markers.
({String text, List<ChatSource> sources}) splitSources(String content) {
  final idx = content.lastIndexOf('\n\n---\nIzvori:');
  if (idx == -1) {
    return (text: stripCitationMarkers(content), sources: const []);
  }

  final text = content.substring(0, idx).trim();
  final block = content.substring(idx + '\n\n---\n'.length).trim();
  final sources = <ChatSource>[];
  final lineRe = RegExp(r'^\[(\d+)\]\s*(.*)$');
  for (final line in block.split('\n')) {
    final trimmed = line.trim();
    final match = lineRe.firstMatch(trimmed);
    if (match != null) {
      sources.add(ChatSource(
        number: int.tryParse(match.group(1) ?? '') ?? sources.length + 1,
        filename: match.group(2)?.trim() ?? trimmed,
        label: match.group(2)?.trim() ?? trimmed,
      ));
    }
  }
  return (text: stripCitationMarkers(text), sources: sources);
}

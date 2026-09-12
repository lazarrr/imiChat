enum MessageRole { user, assistant }

/// A structured source record emitted by the backend in a
/// `data: {"sources":[...]}` SSE event.
///
/// The backend uses a single citation registry for both the inline
/// `[Izvor N]` tags and this list, so [number] always refers to the same
/// source the model cited.
class ChatSource {
  const ChatSource({
    required this.number,
    required this.filename,
    this.page = 0,
    this.url = '',
    this.type = '',
    this.label = '',
  });

  final int number;
  final String filename;
  final int page;
  final String url;
  final String type;
  final String label;

  String get displayLabel {
    if (label.isNotEmpty) return label;
    if (page > 0) return '$filename — strana $page';
    if (url.isNotEmpty) return '$filename — $url';
    return filename;
  }

  Map<String, dynamic> toJson() => {
        'number': number,
        'filename': filename,
        'page': page,
        'url': url,
        'type': type,
        'label': label,
      };

  factory ChatSource.fromJson(Map<String, dynamic> json) => ChatSource(
        number: (json['number'] as num?)?.toInt() ?? 0,
        filename: json['filename'] as String? ?? '',
        page: (json['page'] as num?)?.toInt() ?? 0,
        url: json['url'] as String? ?? '',
        type: json['type'] as String? ?? '',
        label: json['label'] as String? ?? '',
      );
}

class ChatMessage {
  ChatMessage({
    required this.id,
    required this.role,
    required this.content,
    this.streaming = false,
    List<ChatSource>? sources,
    DateTime? timestamp,
  })  : sources = sources ?? [],
        timestamp = timestamp ?? DateTime.now();

  final String id;
  final MessageRole role;
  String content;
  bool streaming;
  List<ChatSource> sources;
  final DateTime timestamp;

  ChatMessage copyWith({String? content, bool? streaming, List<ChatSource>? sources}) {
    return ChatMessage(
      id: id,
      role: role,
      content: content ?? this.content,
      streaming: streaming ?? this.streaming,
      sources: sources ?? this.sources,
      timestamp: timestamp,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'role': role.name,
        'content': content,
        'sources': sources.map((s) => s.toJson()).toList(),
        'timestamp': timestamp.toIso8601String(),
      };

  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
        id: json['id'] as String,
        role: MessageRole.values.byName(json['role'] as String),
        content: json['content'] as String,
        sources: (json['sources'] as List<dynamic>? ?? [])
            .map((e) => ChatSource.fromJson(e as Map<String, dynamic>))
            .toList(),
        timestamp: DateTime.tryParse(json['timestamp'] as String? ?? ''),
      );
}

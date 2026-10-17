/// Who authored a chat message.
enum MessageRole { user, assistant }

/// A single turn in a chat thread — either the user's question or the
/// assistant's answer.
///
/// [content] and [streaming] are mutable on purpose: the assistant message is
/// appended to as soon as it is created, while the backend is still streaming
/// tokens into it.
class ChatMessage {
  ChatMessage({
    required this.id,
    required this.role,
    required this.content,
    this.streaming = false,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  final String id;
  final MessageRole role;
  String content;

  /// True while tokens for this message are still arriving.
  bool streaming;

  final DateTime timestamp;

  bool get isEmpty => content.trim().isEmpty;

  ChatMessage copyWith({String? content, bool? streaming}) => ChatMessage(
        id: id,
        role: role,
        content: content ?? this.content,
        streaming: streaming ?? this.streaming,
        timestamp: timestamp,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'role': role.name,
        'content': content,
        'timestamp': timestamp.toIso8601String(),
      };

  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
        id: json['id'] as String? ?? '',
        role: MessageRole.values.firstWhere(
          (role) => role.name == json['role'],
          orElse: () => MessageRole.assistant,
        ),
        content: json['content'] as String? ?? '',
        timestamp: DateTime.tryParse(json['timestamp'] as String? ?? ''),
      );
}

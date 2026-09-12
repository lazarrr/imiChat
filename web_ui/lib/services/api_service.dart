import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/chat_message.dart';

/// Minimal API client for the FastAPI chatbot backend.
///
/// Uses [http.Client.send] so we can consume the Server-Sent Events
/// stream incrementally — important on Flutter web, where XHR-style
/// `post()` calls buffer the entire body before returning it.
class ApiService {
  ApiService({required this.baseUrl, http.Client? client})
      : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<({bool healthy, int activeSessions})> healthCheck() async {
    try {
      final res = await _client.get(_uri('/health')).timeout(
            const Duration(seconds: 4),
          );
      if (res.statusCode == 200) {
        final body = jsonDecode(res.body) as Map<String, dynamic>;
        return (
          healthy: body['status'] == 'healthy',
          activeSessions: body['active_sessions'] as int? ?? 0,
        );
      }
      return (healthy: false, activeSessions: 0);
    } catch (_) {
      return (healthy: false, activeSessions: 0);
    }
  }

  /// Streams an answer from `POST /chat/stream`.
  ///
  /// Yields [StreamEvent] objects: either a token or an error/termination
  /// marker. [sessionId] enables multi-turn memory on the server.
  Stream<StreamEvent> chatStream({
    required String question,
    String? sessionId,
    String script = 'latinic',
  }) async* {
    final request = http.Request('POST', _uri('/chat/stream'))
      ..headers['Content-Type'] = 'application/json'
      ..headers['Accept'] = 'text/event-stream'
      ..body = jsonEncode({
        'question': question,
        'session_id': ?sessionId,
        'script': script,
      });

    final http.StreamedResponse response;
    try {
      response = await _client.send(request);
    } catch (e) {
      yield StreamEvent.error('Nije moguće povezati se sa serverom: $e');
      return;
    }

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();
      String detail = body;
      try {
        detail = (jsonDecode(body) as Map<String, dynamic>)['detail']
                ?.toString() ??
            body;
      } catch (_) {}
      yield StreamEvent.error('Greška servera (${response.statusCode}): $detail');
      return;
    }

    yield* _parseSse(response.stream);
  }

  Stream<StreamEvent> _parseSse(Stream<List<int>> byteStream) async* {
    final lines = byteStream
        .transform(utf8.decoder)
        .transform(const LineSplitter());

    await for (final line in lines) {
      if (line.isEmpty || !line.startsWith('data:')) continue;
      final payload = line.substring(5).trim();
      if (payload.isEmpty) continue;

      Map<String, dynamic> data;
      try {
        data = jsonDecode(payload) as Map<String, dynamic>;
      } catch (_) {
        continue;
      }

      if (data['done'] == true) {
        yield const StreamEvent.done();
        return;
      }
      if (data.containsKey('error')) {
        yield StreamEvent.error(data['error'].toString());
        return;
      }
      final rawSources = data['sources'];
      if (rawSources is List) {
        final sources = rawSources
            .whereType<Map<String, dynamic>>()
            .map(ChatSource.fromJson)
            .toList();
        yield StreamEvent.sources(sources);
        continue;
      }
      final token = data['token'];
      if (token is String && token.isNotEmpty) {
        yield StreamEvent.token(token);
      }
    }

    yield const StreamEvent.done();
  }

  void dispose() => _client.close();
}

sealed class StreamEvent {
  const StreamEvent();

  const factory StreamEvent.token(String text) = TokenEvent;
  const factory StreamEvent.sources(List<ChatSource> sources) = SourcesEvent;
  const factory StreamEvent.error(String message) = ErrorEvent;
  const factory StreamEvent.done() = DoneEvent;
}

class TokenEvent extends StreamEvent {
  const TokenEvent(this.text);
  final String text;
}

/// Structured citations emitted by the backend right before `done`.
///
/// Text-parsing the trailing `--- Izvori:` block is kept as a fallback in
/// [splitSources] for messages stored before this event existed.
class SourcesEvent extends StreamEvent {
  const SourcesEvent(this.sources);
  final List<ChatSource> sources;
}

class ErrorEvent extends StreamEvent {
  const ErrorEvent(this.message);
  final String message;
}

class DoneEvent extends StreamEvent {
  const DoneEvent();
}

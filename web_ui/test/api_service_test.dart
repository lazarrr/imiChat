import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/services/api_service.dart';

void main() {
  test('chatStream parses SSE tokens, sources, done and health', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    final port = server.port;

    server.listen((req) async {
      if (req.method == 'GET' && req.uri.path == '/health') {
        req.response
          ..statusCode = 200
          ..headers.contentType = ContentType.json
          ..write(jsonEncode({'status': 'healthy', 'active_sessions': 2}));
        await req.response.close();
        return;
      }
      if (req.method == 'POST' && req.uri.path == '/chat/stream') {
        final body = await utf8.decoder.bind(req).join();
        final payload = jsonDecode(body) as Map<String, dynamic>;
        expect(payload['question'], 'Pitanje?');
        expect(payload['session_id'], 'abc123');

        req.response
          ..statusCode = 200
          ..headers.contentType = ContentType(
            'text',
            'event-stream',
            charset: 'utf-8',
          );
        req.response.write('data: ${jsonEncode({'token': 'Zdra'})}\n\n');
        await req.response.flush();
        req.response.write('data: ${jsonEncode({'token': 'vo!'})}\n\n');
        await req.response.flush();
        req.response.write(
          'data: ${jsonEncode({
            'sources': [
              {
                'number': 1,
                'filename': 'informatika.pdf',
                'page': 12,
                'url': '',
                'type': 'pdf',
                'label': 'informatika.pdf — strana 12',
                'chunks': [
                  {'id': 'chunk-1', 'text': 'Moduli su Softversko inženjerstvo i IS.', 'url': 'https://imi.pmf.kg.ac.rs/vesti/1', 'similarity': 0.8734},
                ],
              },
            ],
          })}\n\n',
        );
        await req.response.flush();
        req.response.write('data: {"done":true}\n\n');
        await req.response.close();
        return;
      }
      req.response.statusCode = 404;
      await req.response.close();
    });

    final api = ApiService(baseUrl: 'http://127.0.0.1:$port');

    final events = await api
        .chatStream(question: 'Pitanje?', sessionId: 'abc123')
        .toList();
    expect(events, hasLength(4));
    expect((events[0] as TokenEvent).text, 'Zdra');
    expect((events[1] as TokenEvent).text, 'vo!');
    final sourcesEvent = events[2] as SourcesEvent;
    expect(sourcesEvent.sources, hasLength(1));
    expect(sourcesEvent.sources.first.number, 1);
    expect(sourcesEvent.sources.first.filename, 'informatika.pdf');
    expect(sourcesEvent.sources.first.page, 12);
    expect(sourcesEvent.sources.first.hasContext, isTrue);
    expect(sourcesEvent.sources.first.chunks, hasLength(1));
    expect(sourcesEvent.sources.first.chunks.first.id, 'chunk-1');
    expect(
      sourcesEvent.sources.first.chunks.first.text,
      'Moduli su Softversko inženjerstvo i IS.',
    );
    expect(
      sourcesEvent.sources.first.chunks.first.url,
      'https://imi.pmf.kg.ac.rs/vesti/1',
    );
    expect(sourcesEvent.sources.first.chunks.first.similarity, 0.8734);
    expect(events[3], isA<DoneEvent>());

    final health = await api.healthCheck();
    expect(health.healthy, isTrue);
    expect(health.activeSessions, 2);

    api.dispose();
    await server.close();
  });

  test('chatStream surfaces server errors', () async {
    final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);

    server.listen((req) async {
      req.response
        ..statusCode = 500
        ..headers.contentType = ContentType.json
        ..write(jsonEncode({'detail': 'Model not loaded'}));
      await req.response.close();
    });

    final api = ApiService(baseUrl: 'http://127.0.0.1:${server.port}');
    final events = await api.chatStream(question: 'x').toList();

    expect(events, hasLength(1));
    expect(events.single, isA<ErrorEvent>());
    expect((events.single as ErrorEvent).message, contains('Model not loaded'));

    api.dispose();
    await server.close();
  });
}

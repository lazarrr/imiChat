import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/models/chat_message.dart';
import 'package:uni_chat_web/models/conversation.dart';

void main() {
  test('ChatSource round-trips answer chunks through JSON', () {
    const source = ChatSource(
      number: 1,
      filename: 'a.pdf',
      page: 2,
      label: 'a.pdf — strana 2',
      chunks: [
        SourceChunk(
          id: 'c1',
          text: 'Kontekst odgovora.',
          url: 'https://imi.pmf.kg.ac.rs/vesti/1',
          similarity: 0.8734,
        ),
      ],
    );

    final restored = ChatSource.fromJson(source.toJson());

    expect(restored.hasContext, isTrue);
    expect(restored.chunks, hasLength(1));
    expect(restored.chunks.first.id, 'c1');
    expect(restored.chunks.first.text, 'Kontekst odgovora.');
    expect(restored.chunks.first.url, 'https://imi.pmf.kg.ac.rs/vesti/1');
    expect(restored.chunks.first.host, 'imi.pmf.kg.ac.rs');
    expect(restored.chunks.first.similarity, 0.8734);
  });

  test('ChatSource without chunks has no context', () {
    final source = ChatSource.fromJson({
      'number': 1,
      'filename': 'a.pdf',
      'page': 2,
      'label': 'a.pdf — strana 2',
    });

    expect(source.hasContext, isFalse);
    expect(source.chunks, isEmpty);
  });

  test('stripCitationMarkers removes markers and tidies spacing', () {
    expect(
      stripCitationMarkers(
        'Predmet nosi 6 ESPB [Izvor 1]. Vežbe su obavezne [izvor 2].',
      ),
      'Predmet nosi 6 ESPB. Vežbe su obavezne.',
    );
  });

  test('splitSources strips inline markers when there is no legacy block', () {
    final result = splitSources('Odgovor [Izvor 3].');

    expect(result.text, 'Odgovor.');
    expect(result.sources, isEmpty);
  });

  test('splitSources parses a legacy block and strips markers', () {
    final result = splitSources(
      'Odgovor [Izvor 1].\n\n---\nIzvori:\n[1] a.pdf — strana 2',
    );

    expect(result.text, 'Odgovor.');
    expect(result.sources, hasLength(1));
    expect(result.sources.first.number, 1);
    expect(result.sources.first.label, 'a.pdf — strana 2');
  });
}

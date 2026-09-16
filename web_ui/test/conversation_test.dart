import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/models/conversation.dart';

void main() {
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

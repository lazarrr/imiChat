import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/models/chat_message.dart';
import 'package:uni_chat_web/models/source_citation.dart';
import 'package:uni_chat_web/services/link_launcher.dart';
import 'package:uni_chat_web/widgets/message_bubble.dart';

void main() {
  group('parseSourceCitation', () {
    test('parses the number and label of a local PDF page', () {
      final citation = parseSourceCitation(
        '[1] informatika_oas_2021.pdf — strana 12',
      );

      expect(citation.number, '1');
      expect(citation.label, 'informatika_oas_2021.pdf — strana 12');
      expect(citation.url, isNull);
    });

    test('strips the web URL out of the visible label', () {
      final citation = parseSourceCitation(
        '[2] Osnovi programiranja — https://pmf.kg.ac.rs/studije?tab=1',
      );

      expect(citation.number, '2');
      expect(citation.label, 'Osnovi programiranja');
      expect(citation.url, Uri.parse('https://pmf.kg.ac.rs/studije?tab=1'));
    });

    test('keeps the page suffix before the URL', () {
      final citation = parseSourceCitation(
        '[3] Program — strana 4 — https://pmf.kg.ac.rs/program#page=4',
      );

      expect(citation.label, 'Program — strana 4');
      expect(citation.url, Uri.parse('https://pmf.kg.ac.rs/program#page=4'));
    });

    test('ignores prose punctuation after the URL', () {
      final citation = parseSourceCitation(
        '[4] Uslovi — https://pmf.kg.ac.rs/uslovi.',
      );

      expect(citation.label, 'Uslovi');
      expect(citation.url, Uri.parse('https://pmf.kg.ac.rs/uslovi'));
    });

    test('does not treat non-http links as clickable', () {
      final citation = parseSourceCitation('[5] fajl — file:///etc/passwd');

      expect(citation.url, isNull);
      expect(citation.label, 'fajl — file:///etc/passwd');
    });

    test('falls back to the whole line when there is no [N] prefix', () {
      final citation = parseSourceCitation('skripta.pdf — strana 2');

      expect(citation.number, isNull);
      expect(citation.label, 'skripta.pdf — strana 2');
    });
  });

  group('openExternalLink', () {
    test('opens http(s) links through the injected launcher', () async {
      final opened = <Uri>[];

      final result = await openExternalLink(
        Uri.parse('https://pmf.kg.ac.rs'),
        launcher: (url) async {
          opened.add(url);
          return true;
        },
      );

      expect(result, isTrue);
      expect(opened, [Uri.parse('https://pmf.kg.ac.rs')]);
    });

    test('rejects unsafe or malformed URLs', () async {
      var called = false;
      Future<bool> launcher(Uri url) async {
        called = true;
        return true;
      }

      for (final url in [
        'javascript:alert(1)',
        'file:///etc/passwd',
        'mailto:student@pmf.kg.ac.rs',
      ]) {
        expect(
          await openExternalLink(Uri.parse(url), launcher: launcher),
          isFalse,
          reason: url,
        );
      }
      expect(called, isFalse);
    });

    test('reports a failed launch instead of throwing', () async {
      final result = await openExternalLink(
        Uri.parse('https://pmf.kg.ac.rs'),
        launcher: (url) async => throw Exception('no browser'),
      );

      expect(result, isFalse);
    });

    test('validates markdown hrefs before opening', () async {
      expect(await openExternalHref(null), isFalse);
      expect(await openExternalHref('not a url'), isFalse);
    });
  });

  group('link taps', () {
    Future<void> pumpMessage(
      WidgetTester tester,
      ChatMessage message,
      LinkLauncher launcher,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: MessageBubble(message: message, onOpenLink: launcher),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 350));
    }

    testWidgets('markdown links in an answer open on tap', (tester) async {
      final opened = <Uri>[];
      final message = ChatMessage(
        id: 'm1',
        role: MessageRole.assistant,
        content: 'Više informacija na [sajtu PMF-a](https://pmf.kg.ac.rs).',
      );

      await pumpMessage(tester, message, (url) async {
        opened.add(url);
        return true;
      });

      await tester.tapOnText(find.textRange.ofSubstring('sajtu PMF-a'));
      await tester.pump();

      expect(opened, [Uri.parse('https://pmf.kg.ac.rs')]);
    });

    testWidgets('source chips with a URL open on tap', (tester) async {
      final opened = <Uri>[];
      final message = ChatMessage(
        id: 'm2',
        role: MessageRole.assistant,
        content:
            'Predmet nosi 6 ESPB [Izvor 1].\n\n---\nIzvori:\n'
            '[1] Osnovi programiranja — https://pmf.kg.ac.rs/studije',
      );

      await pumpMessage(tester, message, (url) async {
        opened.add(url);
        return true;
      });

      expect(find.text('Osnovi programiranja'), findsOneWidget);
      await tester.tap(find.text('Osnovi programiranja'));
      await tester.pump();

      expect(opened, [Uri.parse('https://pmf.kg.ac.rs/studije')]);
    });

    testWidgets('source chips without a URL are not interactive', (
      tester,
    ) async {
      final opened = <Uri>[];
      final message = ChatMessage(
        id: 'm3',
        role: MessageRole.assistant,
        content:
            'Odgovor.\n\n---\nIzvori:\n'
            '[1] informatika_oas_2021.pdf — strana 12',
      );

      await pumpMessage(tester, message, (url) async {
        opened.add(url);
        return true;
      });

      await tester.tap(
        find.text('informatika_oas_2021.pdf — strana 12'),
        warnIfMissed: false,
      );
      await tester.pump();

      expect(opened, isEmpty);
    });
  });
}

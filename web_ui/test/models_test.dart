import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/models/chat_message.dart';
import 'package:uni_chat_web/models/conversation.dart';
import 'package:uni_chat_web/widgets/message_bubble.dart';

void main() {
  group('splitSources', () {
    test('keeps an answer without a citation block unchanged', () {
      final result = splitSources('Predmet nosi 6 ESPB [Izvor 1].');

      expect(result.text, 'Predmet nosi 6 ESPB [Izvor 1].');
      expect(result.sources, isEmpty);
    });

    test('separates the trailing citation block from the answer', () {
      final result = splitSources(
        'Predmet nosi 6 ESPB [Izvor 1].\n\n---\nIzvori:\n'
        '[1] informatika_oas_2021.pdf — strana 12\n'
        '[2] Uvod u programiranje.docx',
      );

      expect(result.text, 'Predmet nosi 6 ESPB [Izvor 1].');
      expect(result.sources, [
        '[1] informatika_oas_2021.pdf — strana 12',
        '[2] Uvod u programiranje.docx',
      ]);
    });

    test('ignores blank lines around the citation block', () {
      final result = splitSources('Odgovor.\n\n---\nIzvori:\n\n[1] a.pdf\n\n');

      expect(result.text, 'Odgovor.');
      expect(result.sources, ['[1] a.pdf']);
    });
  });

  group('persistence', () {
    test('conversation with messages round-trips through JSON', () {
      final convo = Conversation(id: 'c1', title: 'ESPB')
        ..messages.add(ChatMessage(
          id: 'm1',
          role: MessageRole.user,
          content: 'Koliko ESPB ima Uvod u programiranje?',
        ))
        ..messages.add(ChatMessage(
          id: 'm2',
          role: MessageRole.assistant,
          content: 'Predmet nosi 6 ESPB.',
        ));

      final restored = Conversation.fromJson(convo.toJson());

      expect(restored.id, 'c1');
      expect(restored.title, 'ESPB');
      expect(restored.messages, hasLength(2));
      expect(restored.messages.first.role, MessageRole.user);
      expect(restored.messages.last.role, MessageRole.assistant);
      expect(restored.messages.last.content, 'Predmet nosi 6 ESPB.');
      expect(restored.messages.last.streaming, isFalse);
    });

    test('unknown roles and missing fields fall back instead of throwing', () {
      final message = ChatMessage.fromJson({
        'id': 'm1',
        'role': 'system',
        'content': 'x',
      });

      expect(message.role, MessageRole.assistant);
      expect(message.timestamp, isA<DateTime>());
    });
  });

  testWidgets('assistant bubble shows the answer and its source chips',
      (tester) async {
    final message = ChatMessage(
      id: 'm1',
      role: MessageRole.assistant,
      content: 'Predmet nosi 6 ESPB [Izvor 1].\n\n---\nIzvori:\n'
          '[1] informatika_oas_2021.pdf — strana 12',
    );

    await tester.pumpWidget(
      MaterialApp(home: Scaffold(body: MessageBubble(message: message))),
    );

    expect(find.text('Izvori'), findsOneWidget);
    expect(find.text('informatika_oas_2021.pdf — strana 12'), findsOneWidget);
    expect(
      find.textContaining('6 ESPB', findRichText: true),
      findsOneWidget,
    );
    // The citation block itself must not leak into the rendered answer.
    expect(
      find.textContaining('---', findRichText: true),
      findsNothing,
    );
  });
}

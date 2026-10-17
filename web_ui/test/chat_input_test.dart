import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/widgets/chat_input.dart';

void main() {
  late List<String> sent;

  Future<void> pumpInput(
    WidgetTester tester, {
    bool isStreaming = false,
  }) async {
    sent = <String>[];

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ChatInput(
            onSend: sent.add,
            onStop: () {},
            isStreaming: isStreaming,
            onNewConversation: () {},
          ),
        ),
      ),
    );
    await tester.pump();
  }

  Future<void> type(WidgetTester tester, String text) async {
    await tester.enterText(find.byType(TextField), text);
    await tester.pump();
  }

  testWidgets('Enter sends the question', (tester) async {
    await pumpInput(tester);
    await type(tester, 'Koji su smerovi na fakultetu?');

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();

    expect(sent, ['Koji su smerovi na fakultetu?']);
  });

  testWidgets('Numpad Enter sends the question', (tester) async {
    await pumpInput(tester);
    await type(tester, 'Koliko ESPB nosi Analiza 1?');

    await tester.sendKeyEvent(LogicalKeyboardKey.numpadEnter);
    await tester.pump();

    expect(sent, ['Koliko ESPB nosi Analiza 1?']);
  });

  testWidgets('Shift+Enter does not send', (tester) async {
    await pumpInput(tester);
    await type(tester, 'prvi red');

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pump();

    expect(sent, isEmpty);
  });

  testWidgets('Shift+Enter inserts a line break at the caret', (tester) async {
    await pumpInput(tester);
    await type(tester, 'prvi red');

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pump();

    final field = tester.widget<TextField>(find.byType(TextField));
    expect(field.controller?.text, 'prvi red\n');
    expect(sent, isEmpty);
  });

  testWidgets('Empty input is not sent', (tester) async {
    await pumpInput(tester);
    await type(tester, '   ');

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();

    expect(sent, isEmpty);
  });

  testWidgets('Input is cleared after a successful send', (tester) async {
    await pumpInput(tester);
    await type(tester, 'Kako se vrši upis?');

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();

    final field = tester.widget<TextField>(find.byType(TextField));
    expect(field.controller?.text, isEmpty);
  });

  testWidgets('Enter does nothing while an answer is streaming',
      (tester) async {
    await pumpInput(tester, isStreaming: true);
    await type(tester, 'Još jedno pitanje');

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();

    expect(sent, isEmpty);
  });

  testWidgets('A multi-line question is sent with its line breaks intact',
      (tester) async {
    await pumpInput(tester);
    await type(tester, 'prvi red');

    await tester.sendKeyDownEvent(LogicalKeyboardKey.shiftLeft);
    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.shiftLeft);
    await tester.pump();

    final field = tester.widget<TextField>(find.byType(TextField));
    field.controller!.text = '${field.controller!.text}drugi red';
    await tester.pump();

    await tester.sendKeyEvent(LogicalKeyboardKey.enter);
    await tester.pump();

    expect(sent, ['prvi red\ndrugi red']);
  });
}

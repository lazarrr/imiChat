import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/models/chat_message.dart';
import 'package:uni_chat_web/screens/home_screen.dart';
import 'package:uni_chat_web/services/api_service.dart';
import 'package:uni_chat_web/state/chat_controller.dart';
import 'package:uni_chat_web/widgets/message_bubble.dart';

void main() {
  Future<void> pumpApp(WidgetTester tester, ChatController controller) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(home: HomeScreen(controller: controller)),
    );
    await tester.pump();
  }

  testWidgets('Renders welcome view and sidebar', (tester) async {
    final controller = ChatController(
      api: ApiService(baseUrl: 'http://localhost:8000'),
    );

    await pumpApp(tester, controller);

    expect(find.text('UniChat'), findsWidgets);
    expect(find.text('Novi razgovor'), findsOneWidget);
    expect(find.text('Dobrodošli u UniChat'), findsOneWidget);

    controller.dispose();
  });

  testWidgets('New conversation is created and empty', (tester) async {
    final controller = ChatController(
      api: ApiService(baseUrl: 'http://localhost:8000'),
    );

    await pumpApp(tester, controller);

    await tester.tap(find.text('Novi razgovor'));
    await tester.pump();

    expect(controller.conversations, hasLength(1));
    expect(controller.hasActiveMessages, isFalse);
    expect(find.text('Dobrodošli u UniChat'), findsOneWidget);

    controller.dispose();
  });

  testWidgets('Source tile expands to reveal the answer context', (
    tester,
  ) async {
    final message = ChatMessage(
      id: 'm1',
      role: MessageRole.assistant,
      content: 'Odgovor [Izvor 1].',
      sources: const [
        ChatSource(
          number: 1,
          filename: 'a.pdf',
          page: 2,
          label: 'a.pdf — strana 2',
          chunks: [
            SourceChunk(
              id: 'c1',
              text: 'Tačan kontekst odgovora.',
              url: 'https://imi.pmf.kg.ac.rs/vesti/1',
              similarity: 0.8734,
            ),
          ],
        ),
      ],
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: MessageBubble(message: message)),
      ),
    );

    expect(find.text('Prikaži kontekst odgovora'), findsOneWidget);
    expect(find.text('Tačan kontekst odgovora.'), findsNothing);

    await tester.tap(find.text('Prikaži kontekst odgovora'));
    await tester.pumpAndSettle();

    expect(find.text('Sakrij kontekst odgovora'), findsOneWidget);
    expect(find.text('Tačan kontekst odgovora.'), findsOneWidget);
    expect(find.text('sličnost 87%'), findsOneWidget);
    expect(find.text('imi.pmf.kg.ac.rs'), findsOneWidget);
  });
}

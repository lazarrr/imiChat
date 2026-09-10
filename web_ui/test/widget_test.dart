import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:uni_chat_web/screens/home_screen.dart';
import 'package:uni_chat_web/services/api_service.dart';
import 'package:uni_chat_web/state/chat_controller.dart';

void main() {
  Future<void> pumpApp(WidgetTester tester, ChatController controller) async {
    tester.view.physicalSize = const Size(1440, 900);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      MaterialApp(
        home: HomeScreen(controller: controller),
      ),
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
}

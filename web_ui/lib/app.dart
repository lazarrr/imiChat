import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/api_service.dart';
import 'state/chat_controller.dart';
import 'theme.dart';

class UniChatApp extends StatefulWidget {
  const UniChatApp({super.key, required this.apiBaseUrl});

  final String apiBaseUrl;

  @override
  State<UniChatApp> createState() => _UniChatAppState();
}

class _UniChatAppState extends State<UniChatApp> {
  late final ApiService _api;
  late final ChatController _controller;

  @override
  void initState() {
    super.initState();
    _api = ApiService(baseUrl: widget.apiBaseUrl);
    _controller = ChatController(api: _api);
  }

  @override
  void dispose() {
    _controller.dispose();
    _api.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _controller,
      builder: (context, _) {
        return MaterialApp(
          title: 'UniChat — Univerzitetski asistent',
          debugShowCheckedModeBanner: false,
          theme: buildTheme(Brightness.light),
          darkTheme: buildTheme(Brightness.dark),
          themeMode: _controller.themeMode,
          themeAnimationDuration: const Duration(milliseconds: 280),
          themeAnimationCurve: kEase,
          scrollBehavior: const MaterialScrollBehavior().copyWith(
            scrollbars: true,
          ),
          home: HomeScreen(controller: _controller),
        );
      },
    );
  }
}
